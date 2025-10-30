from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Callable, Optional


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


def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
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
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


def SamplerQNN() -> nn.Module:
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return softmax(self.net(inputs), dim=-1)

    return SamplerModule()


class ClassicalRBF(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class HybridKernelMethod(nn.Module):
    """
    Hybrid kernel that blends a classical RBF kernel with a quantum‑encoded kernel.
    An optional neural‑network feature extractor (e.g. fraud‑detection style) can
    transform the raw inputs before similarity is computed.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        feature_extractor: Optional[nn.Module] = None,
        use_quantum: bool = False,
        quantum_kernel: Optional[nn.Module] = None,
        weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.classical_kernel = ClassicalRBF(gamma)
        self.feature_extractor = feature_extractor
        self.use_quantum = use_quantum
        self.quantum_kernel = quantum_kernel
        self.weight = float(weight)
        if self.use_quantum and self.quantum_kernel is None:
            raise ValueError("Quantum kernel must be provided when use_quantum=True")

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x) if self.feature_extractor else x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._extract(x)
        y = self._extract(y)
        classical = self.classical_kernel(x, y).squeeze(-1)
        if self.use_quantum:
            quantum = self.quantum_kernel(x, y).squeeze(-1)
            return self.weight * classical + (1.0 - self.weight) * quantum
        return classical

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        self.eval()
        return np.array([[self.forward(x, y).item() for y in b] for x in a])


class FastHybridEstimator:
    """
    Lightweight evaluator that applies the hybrid kernel across many parameter sets
    with optional shot noise to mimic quantum sampling uncertainty.
    """

    def __init__(self, kernel: HybridKernelMethod) -> None:
        self.kernel = kernel

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        rng = np.random.default_rng(seed)
        results: List[List[float]] = []
        self.kernel.eval()
        with torch.no_grad():
            for params in parameter_sets:
                x = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                y = x.clone()
                value = self.kernel(x, y).item()
                if shots is None:
                    results.append([value])
                else:
                    noisy = rng.normal(value, max(1e-6, 1.0 / shots))
                    results.append([float(noisy)])
        return results


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "SamplerQNN",
    "ClassicalRBF",
    "HybridKernelMethod",
    "FastHybridEstimator",
]
