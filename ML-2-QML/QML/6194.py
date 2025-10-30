"""Enhanced quantum circuit factory for the incremental data‑uploading classifier."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import CXGate


class QuantumClassifierModel:
    """
    Variational quantum circuit engineered for hybrid QML experiments.
    The ansatz consists of *depth* layers of parameterised rotations (Rx,Rz)
    followed by a full‑chain of CX gates to entangle all qubits.  The circuit
    is built with explicit encoding parameters and a weight vector for the
    variational parameters.

    The :meth:`build_classifier_circuit` static method returns:
        - ``circuit``: ``qiskit.QuantumCircuit`` ready for simulation or execution.
        - ``encoding``: list of encoding parameters (one per qubit).
        - ``weights``: list of variational parameters.
        - ``observables``: list of SparsePauliOp objects representing Z‑measurements
          on each qubit, which map to the classical logits.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the variational circuit.
    depth : int
        Number of rotation‑entanglement blocks.

    Returns
    -------
    Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]
        The circuit and associated meta‑data.
    """

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int,
                                 **kwargs) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        # Encoding parameters (data‑dependent rotations)
        encoding = ParameterVector("x", num_qubits)

        # Variational parameters
        weights = ParameterVector("theta", num_qubits * depth * 2)  # Rx + Rz per qubit per layer

        circuit = QuantumCircuit(num_qubits)

        # Data‑encoding layer
        for qubit, param in enumerate(encoding):
            circuit.rx(param, qubit)

        weight_index = 0
        for _ in range(depth):
            # Parameterised rotations
            for qubit in range(num_qubits):
                circuit.ry(weights[weight_index], qubit)
                weight_index += 1
                circuit.rz(weights[weight_index], qubit)
                weight_index += 1

            # Full‑chain CX entanglement
            for qubit in range(num_qubits - 1):
                circuit.cx(qubit, qubit + 1)
            # Optional wrap‑around entanglement for circular topology
            circuit.cx(num_qubits - 1, 0)

        # Observables: Z measurement on each qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                       for i in range(num_qubits)]

        return circuit, list(encoding), list(weights), observables


__all__ = ["QuantumClassifierModel"]
