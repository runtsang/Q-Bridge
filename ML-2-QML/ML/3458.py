"""Unified classical regression module with quantum feature extraction and graph‑based regularisation.

The API mirrors the original ``QModel`` and ``RegressionDataset`` so that existing
training scripts remain unchanged.  The model consists of three parts:

* **Classical backbone** – a feed‑forward network that maps the input features to a scalar.
* **Quantum feature extractor** – a variational circuit that encodes each sample
  into a state and measures Pauli‑Z expectations.
* **Graph regulariser** – builds a fidelity‑thresholded graph from the quantum
  features and adds a Laplacian penalty to the loss, encouraging predictions of
  similar samples to be close.

Only standard scientific Python packages are required
(``numpy``, ``torch``, ``torchquantum``, ``networkx``).
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Iterable, List, Tuple
import networkx as nx
import torchquantum as tq

# --------------------------------------------------------------------------- #
# 1. Data generation – identical to the seed with optional noise
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int,
    samples: int,
    *,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return features and labels for a synthetic regression task.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature vector.
    samples : int
        Number of samples to generate.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the labels.

    Returns
    -------
    features : np.ndarray, shape (samples, num_features)
    labels   : np.ndarray, shape (samples,)
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std > 0.0:
        y += np.random.normal(scale=noise_std, size=y.shape)
    return x, y.astype(np.float32)


# --------------------------------------------------------------------------- #
# 2. Dataset
# --------------------------------------------------------------------------- #
class UnifiedRegressionDataset(Dataset):
    """
    Dataset that returns a feature vector and a regression target.
    """

    def __init__(self, samples: int, num_features: int, noise_std: float = 0.0):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise_std=noise_std
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# 3. Helper: state fidelity for vectors
# --------------------------------------------------------------------------- #
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Return the absolute square of the inner product between two vectors.
    """
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


# --------------------------------------------------------------------------- #
# 4. Quantum feature extractor
# --------------------------------------------------------------------------- #
class QuantumFeatureExtractor(tq.QuantumModule):
    """
    Variational circuit that encodes a classical vector into a quantum state
    and measures Pauli‑Z expectation values.
    """

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps each feature to a Ry rotation
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"],
            parametric=True,
        )
        self.random_layer = tq.RandomLayer(
            n_ops=30,
            wires=list(range(num_wires)),
            has_params=True,
        )
        # Measure all qubits in the computational basis
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev)
        self.random_layer(qdev)
        return self.measure(qdev)


# --------------------------------------------------------------------------- #
# 5. Unified model
# --------------------------------------------------------------------------- #
class UnifiedQModel(nn.Module):
    """
    Hybrid model: classical backbone + quantum feature extractor + graph regulariser.
    """

    def __init__(
        self,
        num_features: int,
        num_wires: int,
        hidden_sizes: List[int] = [32, 16],
        graph_threshold: float = 0.9,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_wires = num_wires
        self.graph_threshold = graph_threshold

        # Classical backbone
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.classical = nn.Sequential(*layers)

        # Quantum feature extractor
        self.quantum = QuantumFeatureExtractor(num_wires)

    def forward(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor, shape (batch, num_features)

        Returns
        -------
        preds   : torch.Tensor, shape (batch,)
        penalty : torch.Tensor, scalar penalty term from the Laplacian
        """
        # Classical prediction
        preds = self.classical(state_batch).squeeze(-1)

        # Quantum feature extraction
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.num_wires,
            bsz=bsz,
            device=state_batch.device,
        )
        q_features = self.quantum(qdev)  # shape (batch, num_wires)

        # Graph Laplacian penalty
        with torch.no_grad():
            # Normalize features for fidelity calculation
            norm_feats = F.normalize(q_features, dim=1, p=2)
            # Pairwise fidelity matrix
            fid_mat = torch.matmul(norm_feats, norm_feats.t()).clamp(0.0, 1.0)
            # Build adjacency (binary)
            adj = torch.where(fid_mat >= self.graph_threshold, torch.ones_like(fid_mat), torch.zeros_like(fid_mat))
            # Laplacian
            deg = torch.diag(torch.sum(adj, dim=1))
            lap = deg - adj
            # Penalty: preds^T L preds
            penalty = torch.trace(preds.t() @ lap @ preds)
        return preds, penalty


__all__ = [
    "generate_superposition_data",
    "UnifiedRegressionDataset",
    "state_fidelity",
    "QuantumFeatureExtractor",
    "UnifiedQModel",
]
