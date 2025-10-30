import math
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class MultiHeadAttentionQuantum(nn.Module):
    """Multi‑head attention where the projections are refined by a tiny quantum circuit."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_layer = self._build_q_layer(self.d_k)

    def _build_q_layer(self, n_wires: int) -> nn.Module:
        class QLayer(tq.QuantumModule):
            def __init__(self) -> None:
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)]
                )
                self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(qdev, x)
                for gate, wire in zip(self.parameters, range(self.n_wires)):
                    gate(qdev, wires=wire)
                for wire in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[wire, wire + 1])
                tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
                return self.measure(qdev)
        return QLayer()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        k = self.k_linear(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(2, 3)
        q = self.q_linear(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(2, 3)
        v = self.v_linear(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(2, 3)
        # Apply quantum module head‑wise
        q = self._apply_quantum(q)
        k = self._apply_quantum(k)
        v = self._apply_quantum(v)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(2, 3).contiguous().view(batch, seq_len, -1)
        return self.combine(attn_out)

    def _apply_quantum(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, heads, dim = tensor.shape
        outputs = []
        for b in range(batch):
            batch_out = []
            for h in range(heads):
                head_vec = tensor[b, :, h, :].unsqueeze(0)  # (1, seq_len, dim)
                qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=head_vec.size(0), device=head_vec.device)
                out = self.q_layer(head_vec, qdev)
                batch_out.append(out)
            outputs.append(torch.stack(batch_out, dim=1))
        return torch.cat(outputs, dim=0)


class FeedForwardQuantum(nn.Module):
    """Feed‑forward that maps each token through a small quantum circuit before linear projection."""

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self._build_q_layer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=use_bias)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=use_bias)

    def _build_q_layer(self, n_qubits: int) -> nn.Module:
        class QLayer(tq.QuantumModule):
            def __init__(self) -> None:
                super().__init__()
                self.n_wires = n_qubits
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)]
                )
                self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(self.n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(qdev, x)
                for gate, wire in zip(self.parameters, range(self.n_wires)):
                    gate(qdev, wires=wire)
                return self.measure(qdev)
        return QLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=token.size(0), device=token.device)
            inp = token[:, :self.q_layer.n_wires]
            out = self.q_layer(inp, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_attn: int, n_qubits_ffn: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, use_bias)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout, use_bias)
        else:
            self.ffn = FeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding compatible with batch_first=True."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TextClassifierQuantum(nn.Module):
    """Transformer‑based classifier with optional quantum layers."""

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_blocks: int,
                 ffn_dim: int, num_classes: int, dropout: float = 0.1,
                 n_qubits_attn: int = 0, n_qubits_ffn: int = 0,
                 use_bias: bool = False) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        if n_qubits_attn > 0 or n_qubits_ffn > 0:
            self.transformer = nn.Sequential(
                *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                          n_qubits_attn, n_qubits_ffn, dropout, use_bias)
                  for _ in range(num_blocks)]
            )
        else:
            self.transformer = nn.Sequential(
                *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                  for _ in range(num_blocks)]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised Qiskit circuit."""

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


__all__ = [
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifierQuantum",
    "FastBaseEstimator",
]
