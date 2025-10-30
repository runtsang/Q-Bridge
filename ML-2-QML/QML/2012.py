"""Quantum classifier module with advanced variational ansatz and simulation utilities.

The module extends the original circuit builder with a hardware‑efficient ansatz,
entangling layers, and a helper to compute expectation values via a QASM simulator.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np


class QuantumClassifierModel:
    """Wrapper that builds a variational circuit and exposes simulation helpers."""

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[
        QuantumCircuit,
        List[ParameterVector],
        List[ParameterVector],
        List[SparsePauliOp],
    ]:
        """
        Construct a hardware‑efficient ansatz with data re‑uploading and entangling blocks.

        Parameters
        ----------
        num_qubits : int
            Number of qubits (features).
        depth : int
            Number of variational layers.

        Returns
        -------
        Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]
            The circuit, encoding parameters, variational parameters, and Z observables.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        qc = QuantumCircuit(num_qubits)

        # Data encoding
        for i, param in enumerate(encoding):
            qc.rx(param, i)

        # Variational layers
        idx = 0
        for _ in range(depth):
            # Single‑qubit rotations
            for i in range(num_qubits):
                qc.ry(weights[idx], i)
                idx += 1
            # Entangling block
            for i in range(num_qubits - 1):
                qc.cz(i, i + 1)
            # Mirror entangling for hardware efficiency
            for i in range(num_qubits - 1, 0, -1):
                qc.cz(i, i - 1)

        # Observables: Z on each qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        return qc, [encoding], [weights], observables

    @staticmethod
    def simulate_expectation(
        circuit: QuantumCircuit,
        params: dict,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Compute expectation values of the Z observables for the circuit.

        Parameters
        ----------
        circuit : QuantumCircuit
            The variational circuit.
        params : dict
            Mapping from ParameterVector names to numerical values.
        shots : int, optional
            Number of shots for the simulator.

        Returns
        -------
        np.ndarray
            Expectation values for each qubit.
        """
        from qiskit import Aer, execute

        backend = Aer.get_backend("qasm_simulator")
        bound_circ = circuit.bind_parameters(params)
        job = execute(bound_circ, backend=backend, shots=shots)
        result = job.result()
        counts = result.get_counts(bound_circ)

        expectations = np.zeros(circuit.num_qubits)
        for qubit in range(circuit.num_qubits):
            # Z expectation: (P(+1) - P(-1))
            plus_key = "1" * qubit + "0" * (circuit.num_qubits - qubit - 1)
            minus_key = "0" * qubit + "1" * (circuit.num_qubits - qubit - 1)
            p_plus = counts.get(plus_key, 0) / shots
            p_minus = counts.get(minus_key, 0) / shots
            expectations[qubit] = p_plus - p_minus
        return expectations


# Backwards‑compatibility wrapper
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[
    QuantumCircuit,
    List[ParameterVector],
    List[ParameterVector],
    List[SparsePauliOp],
]:
    """Delegate to ``QuantumClassifierModel.build_classifier_circuit``."""
    return QuantumClassifierModel.build_classifier_circuit(num_qubits, depth)


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
