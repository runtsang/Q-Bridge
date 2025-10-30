import torch
import torch.nn as nn
import networkx as nx
from typing import Tuple

class ClassicalRBFKernel(nn.Module):
    """Classical RBF kernel used for feature expansion."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim), prototypes: (n_protos, dim)
        diff = x.unsqueeze(1) - prototypes.unsqueeze(0)  # (batch, n_protos, dim)
        return torch.exp(-self.gamma * (diff ** 2).sum(dim=2))

class HybridFCL(nn.Module):
    """
    Hybrid fully‑connected layer that merges kernel expansion,
    graph‑based regularisation and optional sequential processing.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_prototypes: int = 64,
                 kernel_gamma: float = 1.0,
                 use_quantum: bool = False,
                 n_qubits: int = 4,
                 seq: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_quantum = use_quantum
        self.seq = seq

        # Prototype vectors for kernel expansion
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, input_dim))

        # Kernel module
        if use_quantum:
            # Quantum kernel placeholder – will be overridden in qml_code
            self.kernel = None
        else:
            self.kernel = ClassicalRBFKernel(gamma=kernel_gamma)

        # Linear mapping from kernel feature to hidden dimension
        self.linear = nn.Linear(n_prototypes, hidden_dim)

        # Optional sequence processing
        if seq:
            if use_quantum:
                # Quantum LSTM placeholder – will be overridden in qml_code
                self.lstm = None
            else:
                self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for both static and sequential inputs.
        For static inputs: x shape (batch, input_dim)
        For sequences: x shape (batch, seq_len, input_dim)
        """
        if self.seq:
            b, t, d = x.shape
            x_flat = x.reshape(b * t, d)
            feats = self.kernel(x_flat, self.prototypes)  # (b*t, n_protos)
            feats = feats.reshape(b, t, -1)
            feats = self.linear(feats)  # (b, t, hidden_dim)
            out, _ = self.lstm(feats)
            return out
        else:
            feats = self.kernel(x, self.prototypes)  # (batch, n_protos)
            feats = self.linear(feats)  # (batch, hidden_dim)
            return feats

    def compute_adjacency(self,
                          states: torch.Tensor,
                          threshold: float = 0.8,
                          secondary: float | None = None,
                          secondary_weight: float = 0.5) -> nx.Graph:
        """
        Build a weighted graph from the similarity of hidden states.
        """
        graph = nx.Graph()
        n = states.shape[0]
        graph.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if self.use_quantum:
                    fid = torch.abs(torch.dot(states[i], states[j])) ** 2
                else:
                    fid = torch.dot(states[i], states[j]).item() ** 2
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = ["HybridFCL"]
