"""Quantum hybrid graph‑transformer architecture using TorchQuantum and Qiskit."""

from __future__ import annotations

import math
import itertools
from typing import Sequence, Iterable, Callable, List

import torch
import torch.nn as nn
import torch.quantum as tq
import torch.quantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
import qutip as qt
import networkx as nx

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class QuantumLayer(tq.QuantumModule):
    """Simple quantum encoder that applies trainable rotations and a CNOT chain."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev, x)
        for gate in self.parameters:
            gate(qdev)
        return self.measure(qdev)


class MultiHeadAttentionQuantum(nn.Module):
    """Quantum‑enhanced multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: tq.QuantumDevice | None = None):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.q_layer = QuantumLayer(self.d_k)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.d_k)
        self.combine = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _quantum_project(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, embed_dim = x.shape
        x_head = x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        flat = x_head.view(-1, self.d_k)
        qdev = self.q_device.copy(bsz=flat.size(0), device=flat.device)
        out = self.q_layer(flat, qdev)
        out = out.view(batch, self.num_heads, seq_len, self.d_k)
        return out.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self._quantum_project(x)
        k = self._quantum_project(x)
        v = self._quantum_project(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        return self.combine(out)


class FeedForwardQuantum(nn.Module):
    """Feed‑forward network realised by a quantum module."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.q_layer = QuantumLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        flat = x.view(-1, x.size(-1))
        qdev = self.q_device.copy(bsz=flat.size(0), device=flat.device)
        out = self.q_layer(flat, qdev)
        out = self.linear1(self.dropout(out))
        out = self.linear2(F.relu(out))
        return out.view(batch, seq_len, -1)


class TransformerBlockQuantum(nn.Module):
    """Quantum transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_tf: int, n_qubits_ffn: int,
                 dropout: float = 0.1, q_device: tq.QuantumDevice | None = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


def quantum_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Absolute squared overlap between pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency_q(quantum_states: List[qt.Qobj], threshold: float,
                         secondary: float | None = None,
                         secondary_weight: float = 0.5) -> nx.Graph:
    """Build graph from quantum state fidelities."""
    n = len(quantum_states)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            fid = quantum_fidelity(quantum_states[i], quantum_states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph


class FastBaseEstimatorQuantum:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_vals: Sequence[float]) -> QuantumCircuit:
        mapping = dict(zip(self._params, param_vals))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        results: List[List[complex]] = []
        for params in parameter_sets:
            state = Statevector.from_instruction(self._bind(params))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class HybridGraphTransformerQuantum(nn.Module):
    """Quantum transformer operating on graph‑structured data."""
    def __init__(self,
                 num_nodes: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 n_qubits_tf: int,
                 n_qubits_ffn: int,
                 threshold: float = 0.8,
                 secondary: float | None = None,
                 secondary_weight: float = 0.5,
                 dropout: float = 0.1):
        super().__init__()
        self.node_state = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.transformer = nn.Sequential(*[
            TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                   n_qubits_tf, n_qubits_ffn,
                                   dropout)
            for _ in range(num_blocks)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch: shape (batch, num_nodes) containing indices of nodes
        """
        # gather node states
        x = self.node_state[batch]  # (batch, num_nodes, embed_dim)
        # build adjacency using classical cosine similarity for simplicity
        adjacency = torch.stack([fidelity_adjacency(x[i], self.threshold,
                                                    self.secondary,
                                                    self.secondary_weight)
                                for i in range(batch.size(0))], dim=0)
        mask = torch.stack([torch.tensor([list(adj.adj[v]) for v in range(x.size(1))], dtype=torch.bool)
                            for adj in adjacency], dim=0)
        x = self.transformer(x, mask=mask)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


__all__ = [
    "QuantumLayer",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "fidelity_adjacency_q",
    "FastBaseEstimatorQuantum",
    "HybridGraphTransformerQuantum",
]
