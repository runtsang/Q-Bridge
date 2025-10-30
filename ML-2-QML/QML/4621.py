"""Hybrid quanvolution classifier with a PennyLane variational filter.

The quantum module encodes each 2×2 image patch into a 4‑qubit circuit,
applies a random variational layer, measures Pauli‑Z expectations,
and constructs a fidelity‑based adjacency graph to smooth the outputs.
"""

from __future__ import annotations

import itertools
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

# --- Utility functions (from GraphQNN) ---------------------------------------

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared inner product of two normalised state vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm) ** 2)

def fidelity_adjacency(
    states: list[np.ndarray],
    threshold: float = 0.8,
    *,
    secondary: float | None = 0.5,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --- Quantum filter ---------------------------------------------------------

class HybridQuanvolutionFilter:
    """Variational quanvolution filter implemented with PennyLane."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        graph_threshold: float = 0.8,
        secondary_threshold: float = 0.5,
        device: str = "default.qubit",
    ) -> None:
        self.device = qml.device(device, wires=4)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.graph_threshold = graph_threshold
        self.secondary_threshold = secondary_threshold
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.device, interface="torch")
        def circuit(patch: torch.Tensor) -> torch.Tensor:
            # Encode each pixel into a Ry rotation
            for i, val in enumerate(patch.flatten()):
                qml.RY(val, wires=i)
            # Random variational layer
            for i in range(4):
                qml.RX(np.random.uniform(0, 2 * np.pi), wires=i)
            for i in range(3):
                qml.CNOT(wires=[i, i + 1])
            # Pauli‑Z expectations as output
            return torch.stack(
                [
                    qml.expval(qml.PauliZ(0)),
                    qml.expval(qml.PauliZ(1)),
                    qml.expval(qml.PauliZ(2)),
                    qml.expval(qml.PauliZ(3)),
                ]
            )

        self.circuit = circuit

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        patches = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # patches: (B, C, nH, nW, k, k)
        nH, nW = patches.shape[2], patches.shape[3]
        out = []
        for b in range(B):
            # Only single channel assumed for simplicity
            patch_vals = patches[b].squeeze(0).reshape(-1, self.kernel_size * self.kernel_size)
            patch_out = []
            for patch in patch_vals:
                patch_out.append(self.circuit(patch))
            patch_out = torch.stack(patch_out, dim=0)  # (N, 4)
            # Construct adjacency graph for this batch
            states = patch_out.detach().cpu().numpy()
            graph = fidelity_adjacency(
                states, self.graph_threshold, secondary=self.secondary_threshold
            )
            L = nx.laplacian_matrix(graph).toarray()
            smooth = torch.from_numpy((torch.eye(L.shape[0]) + torch.from_numpy(L)).float())
            smoothed = torch.matmul(patch_out, smooth)  # (N, 4)
            out.append(smoothed)
        out = torch.stack(out)  # (B, N, 4)
        return out.view(B, -1)

# --- Hybrid classifier -------------------------------------------------------

class HybridQuanvolutionClassifier(nn.Module):
    """Hybrid quantum‑classical classifier using the PennyLane filter."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        graph_threshold: float = 0.8,
        secondary_threshold: float = 0.5,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.filter = HybridQuanvolutionFilter(
            in_channels,
            out_channels=4,
            kernel_size=2,
            stride=2,
            graph_threshold=graph_threshold,
            secondary_threshold=secondary_threshold,
            device=device,
        )
        # Feature size: 4 × 14 × 14
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.filter(x)
        logits = self.linear(feats)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionFilter", "HybridQuanvolutionClassifier"]
