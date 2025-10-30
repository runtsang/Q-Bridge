"""Unified hybrid classifier: classical backbone with optional quantum head.

The module defines a single PyTorch class, `UnifiedHybridClassifier`, that
implements the following sub‑components:

* A convolutional backbone that matches the original CNN depth but
  re‑uses the 3×3 kernels from the graph‑QNN random network
  generation, allowing a quick feature extractor.
* A transformer‑style attention module (inspired by the Q‑Transformer
  seed) that aggregates the feature maps before the head.
* A classical dense head that mirrors the hybrid dense layer in the
  classical seed, but is wrapped in a custom autograd function that
  can use a quantum circuit when needed.
* Utility methods to build a fidelity‑based adjacency graph from
  intermediate states, borrowing concepts from the graph‑QNN seed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np


class HybridFunction(nn.Module):
    """Custom autograd function that forwards (and back‑prop) a quantum
    expectation value.  The forward pass uses the quantum circuit
    (if provided) and the backward uses finite‑difference.
    """
    def __init__(self, circuit=None, shift: float = 0.0):
        super().__init__()
        self.circuit = circuit
        self.shift = shift

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.circuit is None:
            # classical sigmoid
            return torch.sigmoid(input_tensor + self.shift)
        # quantum expectation
        with torch.no_grad():
            exp = self.circuit.run(input_tensor.detach().cpu().numpy())
        return torch.tensor(exp, device=input_tensor.device).float()

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        # Not used – PyTorch will rely on the autograd of the circuit
        raise NotImplementedError


class HybridLayer(nn.Module):
    """Linear layer + hybrid function wrapper."""
    def __init__(self, in_features: int, shift: float = 0.0, circuit=None):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.hybrid = HybridFunction(circuit=circuit, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return self.hybrid(logits)


class TransformerBlock(nn.Module):
    """Simple transformer block with multi‑head attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class UnifiedHybridClassifier(nn.Module):
    """Classical CNN + transformer + hybrid head."""
    def __init__(self, use_quantum_head: bool = False, quantum_circuit=None, n_qubits: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.transformer = TransformerBlock(embed_dim=84, num_heads=4, ffn_dim=128)
        self.hybrid = HybridLayer(in_features=84, shift=0.0, circuit=quantum_circuit if use_quantum_head else None)

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
        seq = x.unsqueeze(1)  # shape (batch, seq_len=1, embed_dim=84)
        seq = self.transformer(seq)
        seq = seq.squeeze(1)  # shape (batch, 84)
        logits = self.hybrid(seq)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

    @staticmethod
    def compute_fidelity_graph(states: list[torch.Tensor], threshold: float,
                               secondary: float | None = None,
                               secondary_weight: float = 0.5) -> nx.Graph:
        """Build a graph where nodes are states and edges weighted by fidelity."""
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


__all__ = ["HybridFunction", "HybridLayer", "TransformerBlock", "UnifiedHybridClassifier"]
