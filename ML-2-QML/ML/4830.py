"""Hybrid classical model combining CNN, auxiliary sampler, and optional quantum head.

The class mirrors the original QuantumNAT architecture but is fully classical.
It can be wrapped around a quantum module (`quantum_module`) for hybrid inference
and is suitable for classification (output dimension 4) and regression.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


# --------------------------------------------------------------------------- #
# Helper: sampler network (classical analogue of the quantum sampler)
# --------------------------------------------------------------------------- #
class SamplerQNN(nn.Module):
    """A tiny feed‑forward network that mimics a quantum sampler.

    It receives a 2‑D input and returns a 2‑D probability vector.  The network
    is intentionally shallow so that it can be dropped in as a drop‑in
    replacement for the quantum sampler during hybrid experiments.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


# --------------------------------------------------------------------------- #
# Dataset utilities (classical regression data)
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate data that mimics the quantum superposition used in the QML repo.

    The target is a smooth non‑linear function of the summed input features,
    providing a simple regression benchmark.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapping the synthetic superposition data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Main hybrid model
# --------------------------------------------------------------------------- #
class HybridNATModel(nn.Module):
    """Fully classical counterpart of the Quantum‑NAT architecture.

    Parameters
    ----------
    mode:
        One of ``"classical"``, ``"quantum"``, or ``"hybrid"``.  In classical
        mode the model behaves like a pure CNN + MLP.  In hybrid mode the
        optional ``quantum_module`` is invoked and its output is concatenated
        with the CNN features before the final head.  ``quantum_module`` must
        accept a tensor of shape ``(batch, n_wires)`` and return a tensor of
        the same shape.
    num_classes:
        Output dimensionality for classification.
    quantum_module:
        Optional callable that implements the quantum part.  If ``None`` a
        dummy module that returns zeros is used.
    """

    def __init__(
        self,
        mode: str = "classical",
        num_classes: int = 4,
        quantum_module: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.quantum_module = quantum_module or (lambda x: torch.zeros_like(x))

        # Feature extractor (CNN)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Optional sampler network (used only in classical mode for sanity checks)
        self.sampler = SamplerQNN()

        # Final classifier/regressor head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)

        # Prepare quantum-like features
        if self.mode in ("quantum", "hybrid") and self.quantum_module is not None:
            # Use a slice of the flattened tensor as a stand‑in for the quantum state
            q_in = flat[:, :4]  # first 4 dims
            q_out = self.quantum_module(q_in)
            combined = torch.cat([flat, q_out], dim=1)
        else:
            # In pure classical mode we simply concatenate a dummy 4‑D vector
            dummy = torch.zeros(bsz, 4, device=flat.device)
            combined = torch.cat([flat, dummy], dim=1)

        out = self.fc(combined)
        return self.norm(out)


__all__ = [
    "HybridNATModel",
    "SamplerQNN",
    "RegressionDataset",
    "generate_superposition_data",
]
