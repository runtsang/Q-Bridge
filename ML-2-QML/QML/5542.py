"""Unified hybrid classifier: quantum‑enhanced architecture.

The module defines a single quantum‑ready class, `UnifiedHybridClassifier`,
that mirrors the classical implementation but replaces the dense head
and the feed‑forward part of the transformer with variational quantum
circuits.  The design retains a classical CNN backbone, a transformer
block with a quantum feed‑forward, and a hybrid quantum head,
enabling evaluation on simulators or hardware.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
import numpy as np
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1.  Quantum hybrid function and layer
# --------------------------------------------------------------------------- #
class QuantumHybridFunction(tq.QuantumModule):
    """Quantum expectation value of PauliZ after encoding input."""
    def __init__(self, n_qubits: int, n_layers: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [idx], "func": "rx", "wires": [idx]}
                for idx in range(n_qubits)
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=10, wires=list(range(n_qubits)))
        self.measurer = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # x shape: (batch, n_qubits)
        self.encoder(q_device, x)
        self.random_layer(q_device)
        return self.measurer(q_device)


class QuantumHybridLayer(tq.QuantumModule):
    """Linear layer followed by a quantum hybrid function."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.linear = nn.Linear(n_qubits, 1)
        self.hybrid_func = QuantumHybridFunction(n_qubits=n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_qubits)
        qdev = tq.QuantumDevice(n_wires=self.hybrid_func.n_qubits, bsz=x.size(0), device=x.device)
        exp = self.hybrid_func(x, qdev)
        logits = self.linear(exp)
        return logits


# --------------------------------------------------------------------------- #
# 2.  Quantum transformer block (classical attention + quantum feed‑forward)
# --------------------------------------------------------------------------- #
class QuantumFeedForward(tq.QuantumModule):
    """Feed‑forward network realised by a quantum module."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [idx], "func": "rx", "wires": [idx]}
                    for idx in range(n_qubits)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out = self.q_layer(token, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)  # (batch, seq_len, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(tq.QuantumModule):
    """Transformer block that uses classical multi‑head attention and a quantum feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# 3.  Unified quantum classifier
# --------------------------------------------------------------------------- #
class UnifiedHybridClassifier(tq.QuantumModule):
    """Quantum‑enhanced CNN + transformer + hybrid head."""
    def __init__(self, n_qubits_head: int = 4, n_qubits_ffn: int = 8,
                 num_heads: int = 4, ffn_dim: int = 128):
        super().__init__()
        # Classical convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Quantum transformer block
        self.transformer = TransformerBlockQuantum(embed_dim=84,
                                                  num_heads=num_heads,
                                                  ffn_dim=ffn_dim,
                                                  n_qubits_ffn=n_qubits_ffn)
        # Quantum hybrid head
        self.hybrid = QuantumHybridLayer(n_qubits=n_qubits_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        # Transformer expects a sequence: treat each feature as a token
        seq = x.unsqueeze(1)  # (batch, seq_len=1, embed_dim=84)
        seq = self.transformer(seq)
        seq = seq.squeeze(1)
        logits = self.hybrid(seq)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

    @staticmethod
    def compute_fidelity_graph(states: list[torch.Tensor], threshold: float,
                               secondary: float | None = None,
                               secondary_weight: float = 0.5) -> nx.Graph:
        """Build a graph where nodes are quantum states and edges weighted by fidelity."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, state_i in enumerate(states):
            for j in range(i + 1, len(states)):
                state_j = states[j]
                # normalize states
                norm_i = state_i / (torch.norm(state_i) + 1e-12)
                norm_j = state_j / (torch.norm(state_j) + 1e-12)
                fid = float((norm_i @ norm_j).abs() ** 2)
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = [
    "QuantumHybridFunction",
    "QuantumHybridLayer",
    "QuantumFeedForward",
    "TransformerBlockQuantum",
    "UnifiedHybridClassifier",
]
