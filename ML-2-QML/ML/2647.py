"""Hybrid regression module: classical residual network + quantum feature extractor."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# 1. Dataset: superposition states + fidelity‑based graph filtering
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int,
    samples: int,
    fidelity_threshold: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of superposition states and return the labels.
    The data is filtered by a fidelity‑based graph to remove highly
    correlated samples.  The filtering is performed on the classical
    feature vectors, which are then passed to the quantum encoder.
    """
    # Basic superposition generation
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)

    # Build a fidelity graph over the classical states
    import networkx as nx
    from scipy.spatial.distance import pdist, squareform

    # Compute pairwise fidelity (inner product) on the real vectors
    fid = 1.0 - pdist(x, metric='cosine')
    fid = squareform(fid)

    # Filter out nodes with high fidelity
    G = nx.Graph()
    G.add_nodes_from(range(len(x)))
    for i, j in nx.non_edges(G):
        if fid[i, j] >= fidelity_threshold:
            G.add_edge(i, j)

    # Keep only a subset of nodes that are not too correlated
    keep = [n for n in G.nodes if G.degree(n) < 2]
    return x[keep], y[keep].astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset that returns quantum states and the corresponding target.
    The ``states`` tensor is complex‑valued and suitable for a quantum
    encoder.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(
            num_features, samples
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# 2. Classical residual network
# --------------------------------------------------------------------------- #
class ClassicalResNet(nn.Module):
    """
    A small feed‑forward network that learns the residual between the
    quantum‑encoded features and the true target.
    """
    def __init__(self, in_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# 3. Quantum encoder + hybrid model
# --------------------------------------------------------------------------- #
class QuantumEncoder(nn.Module):
    """
    Variational circuit that uses a random layer followed by trainable
    RX/RY gates.  The encoder is built on top of TorchQuantum
    (tq) and is fully differentiable.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)


class UnifiedQuantumRegression(nn.Module):
    """
    Hybrid model that runs a quantum circuit to produce feature vectors
    and then learns a residual with a classical network.
    """
    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = QuantumEncoder(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        self.residual = ClassicalResNet(num_features, hidden=64)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Quantum part
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)

        # Classical residual learning
        residual = self.residual(state_batch)
        return self.head(features).squeeze(-1) + residual


__all__ = ["UnifiedQuantumRegression", "RegressionDataset", "generate_superposition_data"]
