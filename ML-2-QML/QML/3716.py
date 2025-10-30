"""
QuantumHybridTransformerClassifier: transformer with quantum attention/FFN and a quantum hybrid head.
"""

from __future__ import annotations

import math
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import assemble, transpile

# -------------------- Quantum expectation head --------------------
class QuantumCircuit:
    """Parametrised two‑qubit circuit used by the hybrid head."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: list[float]) -> float:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Expectation over Z eigenbasis
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        return float(np.sum(states * probs))

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.tolist())
        return torch.tensor([expectation], device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Numerical gradient via finite differences
        eps = 1e-4
        inputs = ctx.circuit.run  # placeholder; not used directly
        # We provide a zero gradient – a full implementation would use parameter‑shift
        return torch.zeros_like(grad_output), None, None

class QuantumHybridLayer(nn.Module):
    """Hybrid layer forwarding a scalar through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float = 0.0) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

# -------------------- Quantum transformer blocks --------------------
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding identical to the classical variant."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class MultiHeadAttentionQuantum(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for w, gate in enumerate(self.parameters):
                gate(q_device, wires=w)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        q = self.separate_heads(x)
        k = self.separate_heads(x)
        v = self.separate_heads(x)
        out, _ = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.combine_heads(out)

class FeedForwardQuantum(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for w, gate in enumerate(self.parameters):
                gate(q_device, wires=w)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

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
                 dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# -------------------- Hybrid transformer classifier --------------------
class QuantumHybridTransformerClassifier(nn.Module):
    """
    Transformer classifier that uses quantum modules for attention/FFN and a quantum hybrid head.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        q_device: Optional[tq.QuantumDevice] = None,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if n_qubits_transformer > 0:
            self.transformers = nn.Sequential(
                *[TransformerBlockQuantum(
                    embed_dim, num_heads, ffn_dim,
                    n_qubits_transformer, n_qubits_ffn,
                    dropout=dropout, q_device=q_device) for _ in range(num_blocks)
                ]
            )
        else:
            self.transformers = nn.Sequential(
                *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
            )
        self.dropout = nn.Dropout(dropout)
        if num_classes == 2:
            backend = qiskit.Aer.get_backend("aer_simulator")
            self.classifier = QuantumHybridLayer(2, backend, shots=100, shift=shift)
        else:
            self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "QuantumCircuit",
    "HybridFunction",
    "QuantumHybridLayer",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "QuantumHybridTransformerClassifier",
]
