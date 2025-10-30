"""Hybrid classical kernel module with fraud‑detection feature mapping and
simple regression estimator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional

import torch
from torch import nn
import numpy as np


# --------------------------------------------------------------------------- #
# Fraud‑detection layer utilities (classical analogue)
# --------------------------------------------------------------------------- #
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


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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
# Simple fully‑connected regression estimator
# --------------------------------------------------------------------------- #
def EstimatorQNN() -> nn.Module:
    """Return a small feed‑forward regressor."""
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorNN()


# --------------------------------------------------------------------------- #
# Hybrid kernel class
# --------------------------------------------------------------------------- #
class QuantumKernelMethod:
    """
    Classical kernel engine that supports:
      * Raw RBF kernel
      * Optional fraud‑detection preprocessing
      * Optional EstimatorQNN regression
    """

    def __init__(
        self,
        mode: str = "classical",
        gamma: float = 1.0,
        fraud_input: Optional[FraudLayerParameters] = None,
        fraud_layers: Optional[Iterable[FraudLayerParameters]] = None,
        estimator: Optional[nn.Module] = None,
    ) -> None:
        if mode!= "classical":
            raise ValueError("Only 'classical' mode is supported in this module.")
        self.gamma = gamma
        self.fraud_model: Optional[nn.Module] = None
        if fraud_input is not None:
            self.fraud_model = build_fraud_detection_program(
                fraud_input, fraud_layers or []
            )
        self.estimator = estimator or EstimatorQNN()

    # ------------------------------------------------------------------ #
    # Feature mapping
    # ------------------------------------------------------------------ #
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fraud‑detection feature map if present."""
        if self.fraud_model is None:
            return x
        return self.fraud_model(x)

    # ------------------------------------------------------------------ #
    # Kernel evaluation
    # ------------------------------------------------------------------ #
    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """RBF kernel on transformed features."""
        x_t = self.transform(x)
        y_t = self.transform(y)
        diff = x_t - y_t
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Compute Gram matrix for two datasets."""
        return np.array(
            [[self.kernel(x, y).item() for y in b] for x in a]
        )

    # ------------------------------------------------------------------ #
    # Regression interface
    # ------------------------------------------------------------------ #
    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 200, lr: float = 1e-3) -> None:
        """Train the embedded EstimatorQNN on the kernel‑mapped features."""
        X_k = torch.stack([self.transform(x) for x in X])
        dataset = torch.utils.data.TensorDataset(X_k, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        opt = torch.optim.Adam(self.estimator.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.estimator.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                pred = self.estimator(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict using the trained estimator on kernel‑mapped features."""
        self.estimator.eval()
        with torch.no_grad():
            X_k = torch.stack([self.transform(x) for x in X])
            return self.estimator(X_k)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "EstimatorQNN",
    "QuantumKernelMethod",
]
