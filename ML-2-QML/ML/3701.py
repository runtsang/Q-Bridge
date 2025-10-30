"""Unified fraud detection estimator combining classical and quantum backends."""

import torch
from torch import nn
import numpy as np
from typing import Iterable, List, Sequence, Tuple, Callable
from dataclasses import dataclass


# --- Parameter container ----------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


# --- Utility helpers -------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    """Clip a scalar to [-bound, bound]."""
    return max(-bound, min(value, bound))


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of floats into a 2‑D tensor (batch, 1)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# --- Classical fraud‑layer construction ------------------------------------- #
def _create_classical_layer(
    params: FraudLayerParameters,
    *,
    clip: bool = True,
    device: torch.device | None = None,
) -> nn.Module:
    """Return a layer that reproduces the photonic block used in the seed."""
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

    linear = nn.Linear(2, 2, bias=True)
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


def build_classical_fraud_model(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a PyTorch sequential model that mirrors the photonic architecture."""
    modules: List[nn.Module] = [_create_classical_layer(input_params, clip=False)]
    modules.extend(_create_classical_layer(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --- Unified estimator ------------------------------------------------------- #
class UnifiedFraudEstimator:
    """Evaluate deterministic or noisy outputs for classical fraud models."""

    def __init__(self, model: nn.Module, *, device: torch.device | None = None):
        self.model = model.to(device) if device else model
        self.device = device or torch.device("cpu")

    # ----- Classical evaluation -------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute deterministic outputs for a batch of parameters."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
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
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int = 1_000,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Add Gaussian shot‑noise to deterministic outputs."""
        raw = self.evaluate(observables, parameter_sets)
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    # ----- Static helpers -------------------------------------------------- #
    @staticmethod
    def build_classical_fraud_model(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> nn.Sequential:
        """Convenience wrapper that returns a fully‑constructed PyTorch model."""
        return build_classical_fraud_model(input_params, layers)


__all__ = ["FraudLayerParameters", "build_classical_fraud_model", "UnifiedFraudEstimator"]
