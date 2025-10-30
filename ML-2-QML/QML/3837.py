"""Hybrid quantum classifier that fuses the data‑encoding and variational ansatz
from the incremental quantum classifier with the photonic layer parameters
used in the fraud‑detection example.  The circuit is fully parameterised
with a `ParameterVector` and measured in the computational basis.

The implementation is written in qiskit and can be executed on any
backend that supports the `QuantumCircuit` interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (used as a template)."""
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


def _apply_layer(
    circuit: QuantumCircuit,
    params: FraudLayerParameters,
    *,
    clip: bool,
    param_start: int,
) -> int:
    """
    Apply a photonic‑style layer to a qiskit circuit using rotations and
    entangling gates.  The parameters are mapped to RX/RZ gates; the
    *param_start* index allows us to reuse a single ParameterVector
    across multiple layers.
    """
    # Base rotation (data‑encoding style)
    circuit.rx(params.bs_theta, 0)
    circuit.rx(params.bs_phi, 1)

    # Phase rotations
    for i, phase in enumerate(params.phases):
        circuit.rz(phase, i)

    # Squeezing‑like rotations (using RZ as a stand‑in)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        circuit.rz(_clip(r, 5.0) if clip else r, i)

    # Entangling CZ
    circuit.cz(0, 1)

    # Additional rotations
    for i, phase in enumerate(params.phases):
        circuit.rz(phase, i)

    # Displacement‑like rotations
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        circuit.rz(_clip(r, 5.0) if clip else r, i)

    # Kerr‑like phase shift
    for i, k in enumerate(params.kerr):
        circuit.rz(_clip(k, 1.0) if clip else k, i)

    return param_start + 1  # One parameter consumed per layer (simplified)


def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], list[SparsePauliOp]]:
    """
    Create a Strawberry‑fields‑style quantum program for the hybrid fraud detection model.
    The circuit returns a tuple containing the circuit, encoding parameters,
    variational parameters, and measurement observables.
    """
    num_qubits = 2
    circuit = QuantumCircuit(num_qubits)
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", len(layers))

    # Data encoding
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational layers
    param_index = 0
    for layer in layers:
        param_index = _apply_layer(circuit, layer, clip=True, param_start=param_index)

    # Observables (Z on each qubit)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


class HybridQuantumClassifier:
    """
    Quantum implementation of the hybrid classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit (default 2 for fraud detection).
    depth : int
        Number of stacked photonic‑style layers.
    fraud_params : Iterable[FraudLayerParameters] | None, optional
        Explicit parameters for each layer.  If omitted, the class
        generates a random sequence of parameters.
    """
    def __init__(
        self,
        num_qubits: int,
        depth: int,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        if fraud_params is None:
            import random

            def rand_pair() -> tuple[float, float]:
                return random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)

            fraud_params = [
                FraudLayerParameters(
                    bs_theta=random.uniform(-1.0, 1.0),
                    bs_phi=random.uniform(-1.0, 1.0),
                    phases=rand_pair(),
                    squeeze_r=rand_pair(),
                    squeeze_phi=rand_pair(),
                    displacement_r=rand_pair(),
                    displacement_phi=rand_pair(),
                    kerr=rand_pair(),
                )
                for _ in range(depth)
            ]
        self.fraud_params = list(fraud_params)
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], list[SparsePauliOp]]:
        """Construct the underlying qiskit circuit."""
        # The first layer uses the 'input' parameters; for simplicity we
        # reuse the first FraudLayerParameters object for all layers.
        input_params = self.fraud_params[0]
        return build_fraud_detection_circuit(input_params, self.fraud_params[1:])

    def get_circuit(self) -> QuantumCircuit:
        """Return the raw quantum circuit."""
        return self.circuit

    def get_parameters(self) -> Tuple[Iterable[ParameterVector], Iterable[ParameterVector]]:
        """Return encoding and variational parameter vectors."""
        return self.encoding, self.weights


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_circuit",
    "HybridQuantumClassifier",
]
