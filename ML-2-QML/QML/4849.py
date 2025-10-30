"""Hybrid fraud detection model using a TorchQuantum kernel and a classical neural network.

The quantum kernel encodes two input vectors via a programmable ansatz. The resulting
similarity values are passed to a classical fraud‑network created from
`FraudLayerParameters`.  The module implements a FastEstimator‑style evaluate routine
that can add Gaussian shot noise, mirroring the classical side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Iterable, List, Callable

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# --------------------------------------------------------------------------- #
# Fraud detection network utilities (adapted for quantum interface)
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
) -> torch.nn.Module:
    """
    Builds a linear‑tanh‑post‑process layer.
    The weight matrix is 2×input_dim; the two rows are populated from the first two entries
    of the original photonic parameter set, the rest are zero‑padded.
    """
    weight = torch.zeros(2, input_dim, dtype=torch.float32)
    weight[0, :2] = torch.tensor([params.bs_theta, params.bs_phi], dtype=torch.float32)
    weight[1, :2] = torch.tensor([params.squeeze_r[0], params.squeeze_r[1]], dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = torch.nn.Linear(input_dim, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = torch.nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(torch.nn.Module):
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
) -> torch.nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, input_dim, clip=False)]
    modules.extend(_layer_from_params(layer, input_dim, clip=True) for layer in layers)
    modules.append(torch.nn.Linear(2, 1))
    return torch.nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# Quantum kernel utilities
# --------------------------------------------------------------------------- #

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# FastEstimator utilities for quantum model
# --------------------------------------------------------------------------- #

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, model: tq.QuantumModule):
        self._model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        results: List[List[float]] = []
        for params in parameter_sets:
            outputs = self._model(torch.tensor(params, dtype=torch.float32))
            row = [obs(outputs) for obs in observables]
            results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
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

# --------------------------------------------------------------------------- #
# Hybrid quantum–classical fraud detection model
# --------------------------------------------------------------------------- #

class HybridFraudKernelModel(tq.QuantumModule):
    """Hybrid fraud‑detection pipeline using a quantum kernel and a classical neural network."""
    def __init__(
        self,
        fraud_input_params: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        reference_vectors: Sequence[torch.Tensor],
    ) -> None:
        super().__init__()
        if len(reference_vectors)!= 2:
            raise ValueError("Exactly two reference vectors are required to match the fraud network input.")
        self.reference_vectors = list(reference_vectors)

        # Quantum kernel ansatz (fixed)
        self.q_device = tq.QuantumDevice(n_wires=4)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Classical fraud network built from the same parameter set
        self.fraud_network = build_fraud_detection_network(
            fraud_input_params, fraud_layers, input_dim=len(self.reference_vectors)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute fraud score for a batch of inputs.
        The quantum kernel is evaluated against each of the two fixed reference vectors
        and the resulting two‑dimensional feature vector is fed into the fraud network.
        """
        features = torch.stack(
            [
                torch.stack(
                    [
                        self._kernel_value(x[i].unsqueeze(0), ref.unsqueeze(0))
                        for ref in self.reference_vectors
                    ],
                    dim=0,
                )
                for i in range(x.shape[0])
            ],
            dim=0,
        )
        return self.fraud_network(features).squeeze()

    def _kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        estimator = FastEstimator(self) if shots is not None else FastBaseEstimator(self)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_network",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "FastBaseEstimator",
    "FastEstimator",
    "HybridFraudKernelModel",
]
