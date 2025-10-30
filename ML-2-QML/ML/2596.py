"""Hybrid fraud detection model combining classical neural network and quantum classifier.

The class integrates the photonic-inspired layer construction from the original FraudDetection
seed with the variational circuit and encoding logic from QuantumClassifierModel.  It
provides a unified API that can evaluate the classical network, the quantum circuit,
or a hybrid combination of both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List

import torch
from torch import nn

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.providers.aer import AerSimulator


@dataclass
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


def _clip(value: float, bound: float) -> float:
    """Clamp a value to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Construct a single neural‑layer with optional clipping."""
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


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """Construct a simple variational ansatz with data‑encoding and read‑out."""
    encoding = ParameterVector("x", num_features)
    weights = ParameterVector("theta", num_features * depth)

    circuit = QuantumCircuit(num_features)
    for param, qubit in zip(encoding, range(num_features)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_features):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_features - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_features - i - 1)) for i in range(num_features)]
    return circuit, list(encoding), list(weights), observables


class FraudDetectionHybrid:
    """Hybrid fraud detection model with classical and quantum components."""

    def __init__(self, num_features: int, depth: int, fraud_params: Iterable[FraudLayerParameters], clip: bool = True):
        self.fraud_params = list(fraud_params)
        self.classical_model = build_fraud_detection_program(self.fraud_params[0], self.fraud_params[1:])
        self.quantum_circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_features, depth)
        self.clip = clip
        self.backend = AerSimulator()

    # ------------------------------------------------------------------
    # Classical API
    # ------------------------------------------------------------------
    def forward_classical(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the classical neural network."""
        return self.classical_model(x)

    # ------------------------------------------------------------------
    # Quantum API
    # ------------------------------------------------------------------
    def forward_quantum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input features as circuit parameters, run the variational
        circuit on the Aer simulator, and return the expectation values
        of the predefined observables.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        expectation_values = []
        for i in range(batch_size):
            binding = {param: float(x[i, j]) for j, param in enumerate(self.encoding)}
            bound_circuit = self.quantum_circuit.bind_parameters(binding)
            result = self.backend.run(bound_circuit).result()
            statevector = result.get_statevector(bound_circuit)
            sv = Statevector(statevector)
            evs = [sv.expectation_value(obs).real for obs in self.observables]
            expectation_values.append(evs)

        return torch.tensor(expectation_values, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Hybrid API
    # ------------------------------------------------------------------
    def hybrid_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the classical and quantum outputs via a simple weighted sum.
        The weighting scheme can be tuned for different experimental regimes.
        """
        class_out = self.forward_classical(x)
        quantum_out = self.forward_quantum(x)
        return 0.5 * class_out + 0.5 * quantum_out

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        """Return a dictionary of trainable parameters for both models."""
        return {
            "classical_params": list(self.classical_model.parameters()),
            "quantum_params": [p for p in self.quantum_circuit.parameters()],
        }

    def weight_sizes(self) -> Tuple[List[int], List[int]]:
        """Return the number of trainable parameters in each sub‑model."""
        class_sizes = [p.numel() for p in self.classical_model.parameters()]
        quantum_sizes = [p.numel() for p in self.quantum_circuit.parameters()]
        return class_sizes, quantum_sizes


__all__ = ["FraudDetectionHybrid", "FraudLayerParameters"]
