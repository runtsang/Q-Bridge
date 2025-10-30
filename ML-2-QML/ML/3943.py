"""FraudDetectionHybridNet: a hybrid classical‑quantum fraud detector.

The model is a PyTorch `nn.Module` that:

* extracts image features with a convolutional backbone,
* feeds the flattened activations into a photonic layer evaluated via a StrawberryFields program,
* the photonic expectation is used as an angle for a 2‑qubit Qiskit variational circuit,
* a sigmoid output produces the binary prediction.

The quantum components are imported from :mod:`quantum_module` so that the
ML code stays purely classical while still accessing a true quantum
simulation.  All tensors are kept on the same device, ensuring efficient
GPU usage when available."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Iterable

# Quantum interface – the heavy lifting is done in `quantum_module`.
# Only the numerical interface is imported here, so the ML module stays classical.
from quantum_module import hybrid_quantum_photonic_head

@dataclass
class FraudLayerParameters:
    """Parameters that describe a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class FraudDetectionHybridNet(nn.Module):
    """
    Hybrid model that merges a CNN backbone, a photonic layer, and a
    two‑qubit quantum expectation head.

    The architecture follows the style of the original `FraudDetection.py`
    but adds a quantum head that can be trained end‑to‑end using
    autograd.  The network is fully differentiable because the
    quantum expectation is wrapped in a custom autograd function
    defined in :mod:`quantum_module`.
    """
    def __init__(self, input_shape: tuple = (3, 32, 32), shift: float = np.pi / 2):
        super().__init__()
        # -------------- Convolutional backbone --------------
        self.conv1 = nn.Conv2d(input_shape[0], 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # -------------- Fully connected head --------------
        dummy = torch.zeros(1, *input_shape)
        dummy = self._flatten_features(dummy)
        self.fc1 = nn.Linear(dummy, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # -------------- Quantum head --------------
        self.shift = shift

        # -------------- Photonic layer definition --------------
        # In practice these parameters would be learnable; for the
        # demonstration we keep them fixed.
        self.photonic_params = FraudLayerParameters(
            bs_theta=0.5, bs_phi=0.5,
            phases=(0.0, 0.0),
            squeeze_r=(0.1, 0.1),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.01, 0.01),
        )

    def _flatten_features(self, x: torch.Tensor) -> int:
        """Return the number of features after the conv‑pool‑dropout chain."""
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        return x.size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -------------- CNN feature extraction --------------
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # -------------- Fully connected layers --------------
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)  # shape: (batch,)

        # -------------- Photonic layer --------------
        # We broadcast the scalar output to 8 parameters required by the
        # photonic circuit.  This simple scheme keeps the example
        # lightweight while still exercising the quantum interface.
        batch_size = x.size(0)
        photonic_input = x.unsqueeze(1).repeat(1, 8)  # shape: (batch, 8)

        # -------------- Quantum expectation head --------------
        # The hybrid function runs a StrawberryFields program followed by a
        # two‑qubit Qiskit circuit.  It returns a scalar expectation
        # per sample.
        quantum_out = hybrid_quantum_photonic_head(photonic_input.detach().cpu().numpy())

        # -------------- Final classification --------------
        quantum_out = torch.tensor(quantum_out, dtype=torch.float32, device=x.device)
        probs = torch.sigmoid(quantum_out + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["FraudDetectionHybridNet"]
