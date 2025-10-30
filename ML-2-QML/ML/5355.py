"""
HybridEstimator – Classical implementation.

Combines the lightweight FastBaseEstimator with
fraud‑detection layers, a simple classifier, and RBF kernel
support.  The estimator accepts any torch.nn.Module,
provides optional Gaussian shot noise, and exposes
factory helpers for the common models used in the seed
projects.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from typing import (
    Callable,
    Iterable,
    List,
    Sequence,
    Tuple,
    Union,
)

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

ScalarObservable = Callable[[torch.Tensor], Union[torch.Tensor, float]]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Converts a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# --------------------------------------------------------------------------- #
# Core estimator
# --------------------------------------------------------------------------- #

class HybridEstimator:
    """
    Evaluate a PyTorch model or a kernel function on batches of inputs.

    Parameters
    ----------
    model : nn.Module
        A neural network or a callable that accepts a tensor and returns a
        tensor.  The estimator will run the model in eval mode.
    kernel : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional
        A symmetric kernel function used by :meth:`kernel_matrix`.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.kernel = kernel
        self.model.eval()

    # --------------------------------------------------------------------- #
    # Evaluation
    # --------------------------------------------------------------------- #

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar
            or a tensor that can be reduced to a scalar.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameters for one batch.
        shots : int, optional
            If supplied, Gaussian shot noise with variance 1/shots is added.
        seed : int, optional
            Seed for the random number generator when ``shots`` is used.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    # --------------------------------------------------------------------- #
    # Kernel utilities
    # --------------------------------------------------------------------- #

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two datasets using the stored kernel.
        """
        if self.kernel is None:
            raise ValueError("No kernel function was supplied to the estimator.")
        return np.array([[float(self.kernel(x, y)) for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Factory helpers – fraud detection, classifier, and RBF kernel
# --------------------------------------------------------------------------- #

# Fraud‑detection layer parameters (mirrors the seed)
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

    def __init__(
        self,
        bs_theta: float,
        bs_phi: float,
        phases: Tuple[float, float],
        squeeze_r: Tuple[float, float],
        squeeze_phi: Tuple[float, float],
        displacement_r: Tuple[float, float],
        displacement_phi: Tuple[float, float],
        kerr: Tuple[float, float],
    ) -> None:
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


def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            return outputs * self.scale + self.shift

    return Layer()


def build_fraud_detection_model(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """
    Create a sequential PyTorch model mirroring the fraud‑detection architecture.
    """
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Build a simple feed‑forward classifier with metadata.
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


# Classical RBF kernel (seed)
class RBFKernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = RBFKernel(gamma)
    return np.array([[float(kernel(x, y)) for y in b] for x in a])


__all__ = [
    "HybridEstimator",
    "FraudLayerParameters",
    "build_fraud_detection_model",
    "build_classifier_circuit",
    "RBFKernel",
    "kernel_matrix",
]
