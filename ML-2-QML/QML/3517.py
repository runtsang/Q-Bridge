"""Hybrid quantum circuit mirroring the classical interface.

The circuit implements a data‑encoding ansatz followed by a depth‑controlled
variational block.  Parameter clipping is applied to the variational angles
to keep the training landscape well‑posed, inspired by the photonic
implementation in the fraud‑detection example.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


@dataclass
class FraudLayerParameters:
    """
    Parameters mirroring the photonic layer used in the fraud‑detection
    example.  They are retained only for API compatibility and are not
    directly used in the current implementation.
    """
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to the range [−bound, +bound]."""
    return max(-bound, min(bound, value))


class QuantumClassifierModel:
    """Quantum circuit that matches the interface of the classical model.

    The circuit encodes each qubit with an independent rotation, then
    applies a stack of Ry rotations and CZ entangling gates.  The total
    number of parameters is ``num_qubits * depth``.  Parameters are
    clipped to the interval [−5, 5] to mimic the bounded parameters
    used in the photonic implementation.
    """

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
    ) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """
        Construct a layered ansatz with explicit encoding and variational parameters.

        Parameters
        ----------
        num_qubits
            Number of qubits in the circuit.
        depth
            Number of variational layers.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)

        # Data encoding
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)

        # Variational ansatz
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(_clip(weights[idx], 5.0), qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        # Observables (Z on each qubit)
        observables: List[SparsePauliOp] = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
        ]

        return circuit, list(encoding), list(weights), observables


__all__ = ["FraudLayerParameters", "QuantumClassifierModel"]
