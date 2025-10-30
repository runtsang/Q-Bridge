from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class QuantumKernel(nn.Module):
    """Classical simulation of a 2‑qubit quantum kernel.

    The kernel encodes a 2‑dimensional input into a 4‑dimensional feature
    vector that mimics the measurement statistics of a random 2‑qubit
    circuit.  The unitary is drawn once from the Haar distribution and
    kept fixed, providing a reproducible quantum‑inspired feature map.
    """
    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        mat = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        q, r = np.linalg.qr(mat)
        d = np.diag(r)
        ph = d / np.abs(d)
        self.unitary = torch.tensor(q @ np.diag(ph), dtype=torch.complex64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (batch, 2) → (batch, 4) feature vector."""
        theta0 = x[:, 0]
        theta1 = x[:, 1]
        c0 = torch.cos(theta0 / 2)
        s0 = torch.sin(theta0 / 2)
        c1 = torch.cos(theta1 / 2)
        s1 = torch.sin(theta1 / 2)
        state = torch.stack(
            [
                c0 * c1,
                c0 * s1,
                s0 * c1,
                s0 * s1,
            ],
            dim=1,
        )
        state = state @ self.unitary.conj().t()
        probs = torch.abs(state) ** 2
        return probs

class FraudDetectionNet(nn.Module):
    """Hybrid fraud‑detection model.

    The network first extracts quantum‑inspired features via
    :class:`QuantumKernel`.  These 4‑dimensional features are then
    processed by a small classical MLP that mirrors the structure of
    the photonic fraud‑detection circuit in the original seed.  The
    clipping logic of the photonic implementation is preserved for
    the linear layers that follow the kernel.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.qkernel = QuantumKernel()
        self.classical = nn.Sequential(
            nn.Linear(4, 2),
            nn.Tanh(),
            nn.Linear(2, 2),
            nn.Tanh(),
            nn.Linear(2, 1),
        )
        self._clip_linear(self.classical[0], bound=5.0)
        self._clip_linear(self.classical[2], bound=5.0)

    def _clip_linear(self, linear: nn.Linear, bound: float) -> None:
        with torch.no_grad():
            linear.weight.clamp_(min=-bound, max=bound)
            linear.bias.clamp_(min=-bound, max=bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qfeat = self.qkernel(x)
        return self.classical(qfeat)

__all__ = ["FraudLayerParameters", "QuantumKernel", "FraudDetectionNet"]
