"""Hybrid quantum classifier that blends a layered data‑encoding ansatz with fraud‑detection style parameter clipping.

The circuit is constructed from a sequence of FraudLayerParameters that are mapped to RX/RZ/CZ gates.
A static method ``build_classifier_circuit`` mimics the classical helper and returns the circuit,
encoding parameters, variational weights and measurement observables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

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

def _apply_layer(circuit: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
    """Map a photonic layer onto a qubit circuit with optional clipping."""
    # Data encoding via RX
    for qubit, angle in enumerate([params.bs_theta, params.bs_phi]):
        circuit.rx(angle, qubit)

    # Phase rotations
    for qubit, phase in enumerate(params.phases):
        circuit.rz(phase, qubit)

    # Squeezing analogue: RZ with clipped amplitude
    for qubit, r in enumerate(params.squeeze_r):
        circuit.rz(_clip(r, 5.0), qubit)

    # Displacement analogue: RX with clipped amplitude
    for qubit, r in enumerate(params.displacement_r):
        circuit.rx(_clip(r, 5.0), qubit)

    # Kerr analogue: CZ with clipped amplitude
    for qubit in range(circuit.num_qubits - 1):
        circuit.cz(qubit, qubit + 1)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Create a layered Qiskit circuit that mimics the photonic fraud‑detection stack."""
    qc = QuantumCircuit(2)
    _apply_layer(qc, input_params, clip=False)
    for layer in layers:
        _apply_layer(qc, layer, clip=True)
    return qc

class HybridClassifier:
    """Class that builds a quantum classifier circuit compatible with the classical interface."""

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int, depth: int
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Construct a layered ansatz with data‑encoding, variational parameters and observables."""
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        qc = QuantumCircuit(num_qubits)

        # Data encoding
        for qubit, param in enumerate(encoding):
            qc.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Observables: Z on each qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

        return qc, list(encoding), list(weights), observables

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "HybridClassifier"]
