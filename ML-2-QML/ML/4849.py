"""Hybrid fraud detection model leveraging classical RBF kernel and a neural network.

The class exposes a FastEstimator‑style evaluate routine that can optionally inject Gaussian
shot noise, mirroring the behaviour of the quantum counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Iterable, List, Callable

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Kernel utilities (derived from the original RBF kernel)
# --------------------------------------------------------------------------- #

class KernalAnsatz(nn.Module):
    """RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper around :class:`KernalAnsatz` that normalises inputs."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Fraud detection network utilities (adapted from the photonic analogue)
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

def _layer_from_params(
    params: FraudLayerParameters,
    input_dim: int,
    *,
    clip: bool
) -> nn.Module:
    """
    Builds a single linear‑tanh‑post‑process layer.
    The weight matrix is forced to be 2×input_dim; the two rows are populated
    from the first two entries of the original photonic parameter set, the rest
    are zero‑padded.  This keeps the interface compatible with the seed while
    allowing an arbitrary number of kernel features.
    """
    weight = torch.zeros(2, input_dim, dtype=torch.float32)
    weight[0, :2] = torch.tensor([params.bs_theta, params.bs_phi], dtype=torch.float32)
    weight[1, :2] = torch.tensor([params.squeeze_r[0], params.squeeze_r[1]], dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(input_dim, 2)
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

def build_fraud_detection_network(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    input_dim: int,
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, input_dim, clip=False)]
    modules.extend(_layer_from_params(layer, input_dim, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# FastEstimator utilities (borrowed from the fast primitives)
# --------------------------------------------------------------------------- #

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# --------------------------------------------------------------------------- #
# Hybrid model combining kernel and fraud network
# --------------------------------------------------------------------------- #

class HybridFraudKernelModel(nn.Module):
    """Hybrid fraud‑detection pipeline using a classical kernel and a neural network."""
    def __init__(
        self,
        kernel_gamma: float,
        fraud_input_params: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        reference_vectors: Sequence[torch.Tensor],
    ) -> None:
        super().__init__()
        if len(reference_vectors)!= 2:
            raise ValueError("Exactly two reference vectors are required to match the fraud network input.")
        self.kernel = Kernel(kernel_gamma)
        self.reference_vectors = list(reference_vectors)
        # Build fraud network with input dimension equal to number of reference vectors (2)
        self.fraud_network = build_fraud_detection_network(
            fraud_input_params, fraud_layers, input_dim=len(self.reference_vectors)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute fraud score for a batch of inputs.
        The kernel is evaluated against each of the two fixed reference vectors
        and the resulting two‑dimensional feature vector is fed into the fraud network.
        """
        features = torch.stack(
            [
                torch.stack(
                    [
                        self.kernel(x[i].unsqueeze(0), ref.unsqueeze(0)).squeeze()
                        for ref in self.reference_vectors
                    ],
                    dim=0,
                )
                for i in range(x.shape[0])
            ],
            dim=0,
        )
        return self.fraud_network(features).squeeze()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Convenience wrapper that mimics the FastEstimator API.
        """
        estimator = FastEstimator(self) if shots is not None else FastBaseEstimator(self)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "FraudLayerParameters",
    "build_fraud_detection_network",
    "FastBaseEstimator",
    "FastEstimator",
    "HybridFraudKernelModel",
]
