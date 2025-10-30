"""Hybrid classical estimator that blends fraud‑detection layers, RBF kernel,
and a linear regression head.

The class is compatible with the original EstimatorQNN API but extends it by
adding a preprocessing network inspired by the fraud‑detection example and
a kernel‑based regression head.  The model can be trained with a small
ridge‑regularized least‑squares solver.

Usage
-----
>>> from EstimatorQNN__gen281 import EstimatorQNNGen
>>> X = torch.randn(10, 2)
>>> y = torch.randn(10, 1)
>>> model = EstimatorQNNGen(input_params=FraudLayerParameters(
...     bs_theta=0.1, bs_phi=0.2, phases=(0.3, 0.4),
...     squeeze_r=(0.5, 0.6), squeeze_phi=(0.7, 0.8),
...     displacement_r=(0.9, 1.0), displacement_phi=(1.1, 1.2),
...     kerr=(1.3, 1.4)), layers=(), gamma=0.5)
>>> model.fit(X, y)
>>> preds = model.predict(X)
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Sequence, Iterable

# --------------------------------------------------------------------------- #
# 1. Fraud‑detection style layer definition (classical)
# --------------------------------------------------------------------------- #

class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

    def __init__(self, bs_theta, bs_phi, phases, squeeze_r, squeeze_phi,
                 displacement_r, displacement_phi, kerr):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# 2. RBF kernel module
# --------------------------------------------------------------------------- #

class RBFKernel(nn.Module):
    """Radial‑basis‑function kernel, compatible with torch operations."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# --------------------------------------------------------------------------- #
# 3. Hybrid estimator
# --------------------------------------------------------------------------- #

class EstimatorQNNGen(nn.Module):
    """
    Hybrid classical estimator that:
    1) Preprocesses inputs with a fraud‑detection style network.
    2) Computes an RBF kernel matrix on the transformed features.
    3) Learns a linear regression head with ridge regularisation.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters] = (),
        gamma: float = 1.0,
        ridge: float = 1e-4,
    ) -> None:
        super().__init__()
        self.feature_extractor = build_fraud_detection_program(input_params, layers)
        self.kernel = RBFKernel(gamma)
        self.ridge = ridge
        self.linear = nn.Linear(1, 1, bias=False)  # weights will be set during fit

    def _kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix K_ij = k(x_i, y_j) using the RBF kernel.
        """
        n, m = X.shape[0], Y.shape[0]
        X_exp = X.unsqueeze(1).expand(n, m, -1)
        Y_exp = Y.unsqueeze(0).expand(n, m, -1)
        return self.kernel(X_exp, Y_exp).squeeze(-1)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit the linear head by solving (K + λI)w = y.
        """
        # Preprocess data
        X_feat = self.feature_extractor(X)
        # Compute Gram matrix
        K = self._kernel_matrix(X_feat, X_feat)
        # Closed‑form ridge solution
        n = K.shape[0]
        A = K + self.ridge * torch.eye(n, device=K.device)
        w = torch.linalg.solve(A, y)
        # Store weights as a 1×1 linear layer
        self.linear.weight.data = w.t()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Return predictions for new data."""
        X_feat = self.feature_extractor(X)
        K = self._kernel_matrix(X_feat, X_feat)
        return self.linear(K)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program",
           "RBFKernel", "EstimatorQNNGen"]
