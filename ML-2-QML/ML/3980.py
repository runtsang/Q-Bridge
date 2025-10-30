"""Hybrid classical-quantum estimator merging classical fraud‑detection layers and a Qiskit quantum circuit."""
from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

# Quantum imports
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimator
from qiskit.primitives import StatevectorEstimator


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


def build_quantum_estimator() -> QEstimator:
    """Return a Qiskit EstimatorQNN with one input and one trainable weight."""
    input_params = [Parameter("x0")]
    weight_params = [Parameter("w0")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(input_params[0], 0)
    qc.rx(weight_params[0], 0)
    observable = SparsePauliOp.from_list([("Y", 1)])
    estimator = StatevectorEstimator()
    return QEstimator(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )


class HybridEstimatorQNN(nn.Module):
    """Hybrid classical‑quantum network for regression."""

    def __init__(
        self,
        quantum_estimator: QEstimator,
        classical_layers: nn.Sequential,
    ) -> None:
        super().__init__()
        self.quantum_estimator = quantum_estimator
        self.classical_layers = classical_layers
        self.quantum_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Bind input and trainable weight to the quantum circuit
        param_dict = {
            self.quantum_estimator.input_params[0]: inputs[:, 0],
            self.quantum_estimator.weight_params[0]: self.quantum_weight.item(),
        }
        circuit = self.quantum_estimator.circuit.bind_parameters(param_dict)
        # Evaluate expectation value
        expectation = self.quantum_estimator.estimator.run(
            [circuit], [], [self.quantum_estimator.observables]
        ).result().values
        q_out = torch.tensor(expectation[0], dtype=inputs.dtype, device=inputs.device).view(-1, 1)
        # Classical prediction plus quantum contribution
        return self.classical_layers(inputs) + self.quantum_weight * q_out


def EstimatorQNN() -> HybridEstimatorQNN:
    """Return a hybrid estimator combining classical fraud‑detection layers and a quantum circuit."""
    classical = build_fraud_detection_program(
        input_params=FraudLayerParameters(
            bs_theta=0.1,
            bs_phi=0.2,
            phases=(0.3, 0.4),
            squeeze_r=(0.5, 0.6),
            squeeze_phi=(0.7, 0.8),
            displacement_r=(0.9, 1.0),
            displacement_phi=(1.1, 1.2),
            kerr=(1.3, 1.4),
        ),
        layers=[
            FraudLayerParameters(
                bs_theta=0.15,
                bs_phi=0.25,
                phases=(0.35, 0.45),
                squeeze_r=(0.55, 0.65),
                squeeze_phi=(0.75, 0.85),
                displacement_r=(0.95, 1.05),
                displacement_phi=(1.15, 1.25),
                kerr=(1.35, 1.45),
            )
        ],
    )
    quantum = build_quantum_estimator()
    return HybridEstimatorQNN(quantum_estimator=quantum, classical_layers=classical)


__all__ = [
    "EstimatorQNN",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "build_quantum_estimator",
    "HybridEstimatorQNN",
]
