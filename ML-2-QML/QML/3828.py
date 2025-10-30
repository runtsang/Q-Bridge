import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence
import numpy as np

class MultiHeadAttentionQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 ansatz_depth: int = 1, num_qubits_per_head: int = 4,
                 q_device: tq.QuantumDevice | None = None):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.ansatz_depth = ansatz_depth
        self.num_qubits_per_head = num_qubits_per_head
        self.q_layer = self.QLayer(num_qubits_per_head, ansatz_depth)
        self.q_device = q_device or tq.QuantumDevice(n_wires=num_qubits_per_head)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, depth: int):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            self.encoder = tq.GeneralEncoder([
                {"input_idx": [idx], "func": "rx", "wires": [idx]}
                for idx in range(n_wires)
            ])
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_dev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_dev, x)
            for gate in self.parameters:
                gate(q_dev)
            for i in range(self.n_wires - 1):
                tqf.cnot(q_dev, wires=[i, i + 1])
            return self.measure(q_dev)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device.copy(bsz=head.size(0), device=head.device)
                head_outputs.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_outputs, dim=1))
        return torch.stack(projections, dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.size(0)
        k = self._apply_quantum_heads(x)
        q = self._apply_quantum_heads(x)
        v = self._apply_quantum_heads(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.combine_heads(out)

class FeedForwardQuantum(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, depth: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_qubits, depth)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, depth: int):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            self.encoder = tq.GeneralEncoder([
                {"input_idx": [idx], "func": "rx", "wires": [idx]}
                for idx in range(n_wires)
            ])
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_dev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_dev, x)
            for gate in self.parameters:
                gate(q_dev)
            return self.measure(q_dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_transformer: int, n_qubits_ffn: int,
                 ansatz_depth: int = 1, ffn_depth: int = 1, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout,
                                              ansatz_depth, n_qubits_transformer)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, ffn_depth, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class HybridTextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_blocks: int,
                 ffn_dim: int, num_classes: int, dropout: float = 0.1,
                 n_qubits_transformer: int = 8, n_qubits_ffn: int = 8,
                 ansatz_depth: int = 1, ffn_depth: int = 1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList([TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                                             n_qubits_transformer, n_qubits_ffn,
                                                             ansatz_depth, ffn_depth, dropout)
                                     for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

    def to_qiskit_circuit(self, tokens: torch.Tensor) -> QuantumCircuit:
        n_tokens = tokens.size(1)
        n_wires = max(self.blocks[0].attn.num_qubits_per_head,
                      self.blocks[0].ffn.q_layer.n_wires)
        qc = QuantumCircuit(n_wires)
        for i, token in enumerate(tokens.unbind(dim=1)):
            val = token.item()
            qc.rx(val, i % n_wires)
        for block in self.blocks:
            qc.h(range(n_wires))
            for i in range(n_wires - 1):
                qc.cx(i, i + 1)
        return qc

class FastBaseEstimator:
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
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class FastQuantumEstimator(FastBaseEstimator):
    def __init__(self, circuit: QuantumCircuit):
        super().__init__(circuit)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        return super().evaluate(observables, parameter_sets)

__all__ = [
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "HybridTextClassifier",
    "FastBaseEstimator",
    "FastQuantumEstimator",
]
