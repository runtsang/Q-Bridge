"""Hybrid regression module combining classical feed‑forward and quantum feature extraction.

This module supplies:
- RegressionDataset for generating synthetic data.
- HybridRegressionModel that uses a quantum encoder followed by a classical linear head.
- Graph utilities to build an adjacency graph from state fidelities, enabling graph‑based regularisation or visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import networkx as nx

# --------------------------------------------------------------------------- #
# 1. Dataset
# --------------------------------------------------------------------------- #
def _generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

@dataclass
class RegressionDataset:
    """Simple regression dataset that returns a dictionary with keys ``states`` and ``target``."""
    samples: int
    num_features: int
    features: np.ndarray | None = None
    labels: np.ndarray | None = None

    def __post_init__(self):
        self.features, self.labels = _generate_superposition_data(self.num_features, self.samples)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# 2. Graph utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute squared overlap between two complex state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.abs(torch.dot(a_norm.conj(), b_norm))**2)

def fidelity_adjacency(states: List[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if j <= i:
                continue
            fid = state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# 3. Hybrid quantum‑classical model
# --------------------------------------------------------------------------- #
class HybridRegressionModel(nn.Module):
    """Hybrid classical‑quantum regression model."""
    class QLayer(nn.Module):
        def __init__(self, num_wires: int, random_ops: int = 30):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=random_ops, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        self.num_features = num_features
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["RegressionDataset", "HybridRegressionModel", "state_fidelity", "fidelity_adjacency"]
