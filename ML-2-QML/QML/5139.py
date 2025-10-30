"""Quantum‑enhanced transformer that reuses classical utilities but replaces core
operations with quantum modules."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
import networkx as nx
import qutip as qt
import scipy as sc
import numpy as np


# --------------------------------------------------------------------------- #
# 1. Quantum convolutional pre‑processor (Qiskit)
# --------------------------------------------------------------------------- #
class QuantumConvFilter:
    """2‑D convolutional filter implemented with a Qiskit circuit."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, shots: int = 100) -> None:
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


# --------------------------------------------------------------------------- #
# 2. Quantum self‑attention (Qiskit)
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Self‑attention implemented as a Qiskit circuit."""
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = qiskit.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> qiskit.QuantumCircuit:
        circuit = qiskit.QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict[str, int]:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


# --------------------------------------------------------------------------- #
# 3. Graph utilities (qutip)
# --------------------------------------------------------------------------- #
def state_fidelity_q(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency_q(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            fid = state_fidelity_q(states[i], states[j])
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
    return G


# --------------------------------------------------------------------------- #
# 4. Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
# 5. Quantum multi‑head attention
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(nn.Module):
    """Quantum‑aware multi‑head attention using TorchQuantum."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(8)
                ]
            )
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        q = self._apply_quantum_heads(x)
        k = self._apply_quantum_heads(x)
        v = self._apply_quantum_heads(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0), float("-inf"))
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.combine_heads(out)


# --------------------------------------------------------------------------- #
# 6. Quantum feed‑forward
# --------------------------------------------------------------------------- #
class FeedForwardQuantum(nn.Module):
    """Feed‑forward realized by a quantum module."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# --------------------------------------------------------------------------- #
# 7. Transformer block (quantum)
# --------------------------------------------------------------------------- #
class TransformerBlockQuantum(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = (
            FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
            if n_qubits_ffn > 0
            else FeedForwardClassical(embed_dim, ffn_dim, dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# 8. Hybrid quantum transformer block with graph masking
# --------------------------------------------------------------------------- #
class HybridQuantumTransformerBlock(TransformerBlockQuantum):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        graph_threshold: float = 0.7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, ffn_dim, n_qubits_transformer, n_qubits_ffn, dropout)
        self.graph_threshold = graph_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        # Build adjacency mask from quantum state fidelities of first sample
        states = [x[0, i] for i in range(seq_len)]
        G = fidelity_adjacency_q(states, self.graph_threshold)
        adj = torch.zeros(seq_len, seq_len, device=x.device)
        for i, j in G.edges():
            adj[i, j] = 1
            adj[j, i] = 1
        torch.diagonal(adj).fill_(1)
        attn_mask = (adj == 0)
        attn_out = self.attn(x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# 9. Hybrid quantum transformer model
# --------------------------------------------------------------------------- #
class HybridQuantumTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 8,
        graph_threshold: float = 0.7,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer_layers = nn.ModuleList([
            HybridQuantumTransformerBlock(
                embed_dim,
                num_heads,
                ffn_dim,
                n_qubits_transformer,
                n_qubits_ffn,
                graph_threshold,
                dropout,
            )
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "QuantumConvFilter",
    "QuantumSelfAttention",
    "state_fidelity_q",
    "fidelity_adjacency_q",
    "PositionalEncoder",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "HybridQuantumTransformerBlock",
    "HybridQuantumTransformer",
]
