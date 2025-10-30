from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterable, Sequence, List, Callable
import numpy as np

# ----------------------------------------------------------------------
#  Classical data generation – identical to the original regression seed
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data where the target is a smooth trigonometric function."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# ----------------------------------------------------------------------
#  Photonic‑style classical layer parameters
# ----------------------------------------------------------------------
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
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]],
                          dtype=torch.float32)
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
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Construct a sequential PyTorch model that mirrors the photonic circuit."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
#  Fast estimator utilities (classical side)
# ----------------------------------------------------------------------
class FastBaseEstimator:
    """Evaluate a torch model for a batch of parameter sets and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    row.append(float(val.mean().cpu() if isinstance(val, torch.Tensor) else val))
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Add Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
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

# ----------------------------------------------------------------------
#  Hybrid model: classical backbone + optional Pennylane variational head
# ----------------------------------------------------------------------
class FraudDetectionHybrid(nn.Module):
    """Classical‑quantum hybrid fraud‑detection network.

    The network first processes the 2‑dimensional input through a stack of
    photonic‑style linear layers (mirroring the Strawberry‑Fields example)
    and then feeds the resulting feature vector to a lightweight Pennylane‑style
    variational circuit that produces a scalar output via a Pauli‑Z expectation
    value.  The quantum block is optional and can be dropped to obtain a purely
    classical model.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Sequence[FraudLayerParameters],
        *,
        quantum: bool = True,
    ) -> None:
        super().__init__()
        self.classical = build_fraud_detection_program(input_params, layers)
        self.quantum = quantum
        if quantum:
            import pennylane as qml
            dev = qml.device("default.qubit", wires=2)
            @qml.qnode(dev, interface="torch")
            def circuit(x):
                # encode classical features
                qml.RY(x[0], wires=0)
                qml.RY(x[1], wires=1)
                # variational block
                for i in range(2):
                    qml.RZ(0.1 * i, wires=i)
                    qml.CNOT(wires=[i, (i + 1) % 2])
                return qml.expval(qml.PauliZ(0))
            self.circuit = circuit
            self.head = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.classical(inputs)
        if self.quantum:
            out = self.circuit(x.squeeze(-1))
            out = self.head(out)
            return out
        return x.squeeze(-1)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "generate_superposition_data",
    "FastBaseEstimator",
    "FastEstimator",
    "FraudDetectionHybrid",
]
