"""Hybrid quantum fraud detection and classification circuit.

The quantum implementation mirrors the classical hybrid model:
a photonic‑inspired fraud‑detection sub‑circuit followed by a Qiskit
parameterised ansatz that encodes data and performs a variational
classification step.  The design uses only Qiskit primitives and
exposes the same public API as the original seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a value to a symmetric bound."""
    return max(-bound, min(bound, value))


def _apply_layer(circ: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
    """Append a photonic‑style layer to a quantum circuit."""
    # Phase rotations
    for i, phase in enumerate(params.phases):
        circ.rz(phase, i)

    # Squeezing approximated by RX and RZ
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        circ.rx(_clip(r, 5), i)
        circ.rz(_clip(phi, 5), i)

    # Entanglement (CNOT chain)
    for i in range(circ.num_qubits - 1):
        circ.cx(i, i + 1)

    # Displacement approximated by RX and RZ
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        circ.rx(_clip(r, 5), i)
        circ.rz(_clip(phi, 5), i)

    # Kerr nonlinearity approximated by RZ
    for i, k in enumerate(params.kerr):
        circ.rz(_clip(k, 1), i)


def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Create a fraud‑detection sub‑circuit using photonic‑style gates."""
    circ = QuantumCircuit(2)
    _apply_layer(circ, input_params, clip=False)
    for layer in layers:
        _apply_layer(circ, layer, clip=True)
    return circ


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct the Qiskit variational classifier circuit."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circ = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circ.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circ.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circ.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circ, list(encoding), list(weights), observables


def build_hybrid_circuit(
    num_qubits: int,
    depth: int,
    fraud_params: Iterable[FraudLayerParameters],
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Combine the fraud‑detection sub‑circuit with the variational classifier."""
    if not fraud_params:
        raise ValueError("At least one FraudLayerParameters instance is required.")

    fraud_circ = build_fraud_detection_circuit(fraud_params[0], fraud_params[1:])
    classifier_circ, enc, wts, obs = build_classifier_circuit(num_qubits, depth)

    # Concatenate circuits on the same qubits
    hybrid_circ = fraud_circ + classifier_circ
    return hybrid_circ, enc, wts, obs


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_circuit",
    "build_classifier_circuit",
    "build_hybrid_circuit",
]
