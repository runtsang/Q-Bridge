"""Hybrid classical regression model with optional quantum head.

The module defines ``HybridQModel`` that extends the original classical
regression code by adding a quantum‑to‑classical bottleneck.  The
class can be instantiated in two modes:

* ``mode='classical'`` – only the classical MLP is used.
* ``mode='hybrid'``   – a quantum submodule is appended after the
  feature extractor.  The quantum part can be frozen or fine‑tuned
  independently.

The dataset and data‑generation utilities are identical to the seed
implementation and are kept for compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np

# ----------------------------------------------------------------------
# Data utilities
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a synthetic regression dataset where the target depends on
    a nonlinear combination of the input features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Compatibility wrapper around the original dataset."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ----------------------------------------------------------------------
# Hybrid model
# ----------------------------------------------------------------------
class HybridQModel(nn.Module):
    """
    Hybrid classical–quantum regression model.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature space.
    num_wires : int | None, optional
        Number of qubits used in the quantum submodule.  If ``None`` the
        quantum part is omitted.
    freeze_cls : bool, default=True
        If ``True`` the classical feature extractor is frozen during
        training of the quantum head.
    freeze_qc : bool, default=True
        If ``True`` the quantum submodule is frozen during training
        of the classical head.
    """
    def __init__(
        self,
        num_features: int,
        num_wires: int | None = None,
        freeze_cls: bool = True,
        freeze_qc: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_wires = num_wires

        # Classical feature extractor (3‑layer MLP)
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Classical head
        self.classical_head = nn.Linear(16, 1)

        # Optional quantum head
        self.use_quantum = num_wires is not None
        if self.use_quantum:
            # The quantum submodule is defined in the qml module; we
            # import lazily to avoid heavy dependencies during pure
            # classical training.
            from.qml import QuantumHead  # type: ignore
            self.quantum_head = QuantumHead(num_wires)
            # Freeze controls
            for p in self.feature_extractor.parameters():
                p.requires_grad = not freeze_cls
            for p in self.quantum_head.parameters():
                p.requires_grad = not freeze_qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return classical prediction; if quantum head is present,
        also return quantum prediction as a tuple.
        """
        features = self.feature_extractor(x)
        classical_pred = self.classical_head(features).squeeze(-1)
        if self.use_quantum:
            quantum_pred = self.quantum_head(features)
            return classical_pred, quantum_pred
        return classical_pred

__all__ = ["HybridQModel", "RegressionDataset", "generate_superposition_data"]
