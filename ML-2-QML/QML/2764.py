"""Hybrid quantum classifier that mirrors the classical interface but uses a variational circuit.

The class exposes a `circuit` method returning a Qiskit `QuantumCircuit` and an
`observables` property.  Parameters are clipped to keep the circuit numerically
stable, and the encoding is a product of Rx gates, followed by a depth‑controlled
ansatz of Ry rotations and CZ entanglers.  Fraud‑detection style layers are
encoded using simple rotations to emulate the photonic operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


@dataclass
class FraudLayerParameters:
    """Parameters that describe a single fraud‑detection style layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a value to ``[-bound, bound]``."""
    return max(-bound, min(bound, value))


def _apply_layer(
    circuit: QuantumCircuit, params: FraudLayerParameters, *, clip: bool
) -> None:
    """Encode a fraud‑detection style layer into the circuit using simple rotations."""
    # Beam‑splitter analogue
    circuit.ry(_clip(params.bs_theta, 5.0) if clip else params.bs_theta, 0)
    circuit.ry(_clip(params.bs_phi, 5.0) if clip else params.bs_phi, 1)

    # Phase rotations (simulated by Rz)
    for i, phase in enumerate(params.phases):
        circuit.rz(_clip(phase, 5.0) if clip else phase, i)

    # Squeezing (simulated by Rz)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        circuit.rz(_clip(r, 5.0) if clip else r, i)

    # Entanglement
    circuit.cz(0, 1)

    # Displacement (simulated by Rx)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        circuit.rx(_clip(r, 5.0) if clip else r, i)

    # Kerr (simulated by Rz)
    for i, k in enumerate(params.kerr):
        circuit.rz(_clip(k, 1.0) if clip else k, i)


def build_classifier_circuit(
    num_qubits: int, depth: int, fraud_params: Iterable[FraudLayerParameters]
) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a variational circuit that incorporates fraud‑detection style layers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (must be 2 to match the fraud layers).
    depth : int
        Depth of the variational ansatz.
    fraud_params : iterable
        Sequence of `FraudLayerParameters` used for the circuit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Fraud‑detection style parameters
    for params in fraud_params:
        _apply_layer(circuit, params, clip=True)

    # Observables
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


class HybridClassifier:
    """Quantum‑inspired hybrid classifier that mimics the classical API."""

    def __init__(self, num_qubits: int, depth: int, fraud_params: Iterable[FraudLayerParameters]) -> None:
        self._circuit, self._encoding, self._weights, self._observables = build_classifier_circuit(
            num_qubits, depth, fraud_params
        )

    def circuit(self) -> QuantumCircuit:
        """Return the underlying Qiskit circuit."""
        return self._circuit

    def observables(self) -> list[SparsePauliOp]:
        """Return measurement observables."""
        return self._observables


__all__ = ["HybridClassifier", "FraudLayerParameters", "build_classifier_circuit"]
