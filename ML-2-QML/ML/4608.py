"""Hybrid fraud‑detection model combining classical neural nets, a photonic circuit,
a Qiskit EstimatorQNN, and a TorchQuantum kernel.

The module is self‑contained and only imports the quantum utilities from the
``qml`` sibling module.  All heavy quantum work is delegated to the
``qml`` code; the ML side remains fully classical (PyTorch).

Typical usage:

    from FraudDetection__gen050 import FraudDetectionModel
    model = FraudDetectionModel(
        input_params=input_layer_params,
        layer_params=[...],
        basis_points=train_features,   # optional reference set for kernel
        kernel_gamma=1.0
    )
    preds = model(torch.tensor(features, dtype=torch.float32))

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
import numpy as np

# Quantum utilities are defined in the sibling ``qml`` module.
# They perform photonic circuit construction, Qiskit EstimatorQNN creation,
# and TorchQuantum kernel evaluation.
try:
    from.qml import kernel_matrix, build_photonic_program, build_qiskit_estimator
except Exception:  # pragma: no cover
    # In environments without a package layout the relative import fails.
    # Users should import the qml module directly or adjust the import path.
    raise


@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic layer (mirrors the original seed)."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


class _LayerFromParams(nn.Module):
    """Constructs a classical layer that mirrors the photonic layer."""
    def __init__(self, params: FraudLayerParameters, clip: bool = False) -> None:
        super().__init__()
        weight = torch.tensor(
            [
                [params.bs_theta, params.bs_phi],
                [params.squeeze_r[0], params.squeeze_r[1]],
            ],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params.displacement_r))
        self.register_buffer("shift", torch.tensor(params.displacement_phi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        out = self.activation(self.linear(x))
        return out * self.scale + self.shift


class FraudDetectionModel(nn.Module):
    """Hybrid fraud‑detection model with classical, photonic, Qiskit, and kernel components."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        basis_points: Optional[torch.Tensor] = None,
        kernel_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        # Classical feature extractor – a small feed‑forward network.
        self.classical_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )
        # Photonic circuit builder (kept for compatibility / hardware execution)
        self.input_params = input_params
        self.layer_params = list(layer_params)
        # Quantum kernel parameters
        self.basis_points = basis_points  # shape (n_basis, 2)
        self.kernel_gamma = kernel_gamma

        # Optional linear combiner for kernel features
        if basis_points is not None:
            # Number of kernel features equals number of basis points
            self.kernel_combiner = nn.Linear(len(basis_points), 1, bias=False)

        # Qiskit EstimatorQNN instance (created lazily)
        self._qiskit_estimator = None

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute fraud‑risk score for input batch ``x`` (shape ``[batch, 2]``)."""
        # Classical score
        cls_score = self.classical_net(x)  # shape (batch, 1)

        # Quantum kernel features if basis points are provided
        if self.basis_points is not None:
            # Compute kernel matrix between batch and basis set
            kernel_feats = kernel_matrix(
                [x[i] for i in range(x.size(0))],
                [self.basis_points[j] for j in range(self.basis_points.size(0))],
                gamma=self.kernel_gamma,
            )  # shape (batch, n_basis)
            kernel_feats = torch.tensor(kernel_feats, dtype=torch.float32)
            # Linear combiner
            kernel_score = self.kernel_combiner(kernel_feats)  # shape (batch, 1)
            # Fuse scores – simple weighted sum
            return cls_score + kernel_score
        return cls_score

    def build_photonic_program(self) -> "sf.Program":
        """Return a Strawberry Fields program for the photonic circuit."""
        return build_photonic_program(self.input_params, self.layer_params)

    def get_qiskit_estimator(self):
        """Instantiate and cache a Qiskit EstimatorQNN."""
        if self._qiskit_estimator is None:
            self._qiskit_estimator = build_qiskit_estimator()
        return self._qiskit_estimator

    # ------------------------------------------------------------------
    #  Convenience helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}(layers={len(self.layer_params)}, "
            f"basis_points={self.basis_points is not None})"
        )


__all__ = [
    "FraudLayerParameters",
    "FraudDetectionModel",
]
