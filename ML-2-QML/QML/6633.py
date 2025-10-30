"""
Core circuit factory for the incremental data‑uploading classifier with an extended ansatz.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator


class QuantumClassifierModel:
    """
    Factory for a variational circuit that mirrors the classical MLP.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Number of variational layers.
    rotation_bases : List[str]
        Rotation gates to use in each layer (e.g., ['rx', 'ry', 'rz']).
    entanglement : str
        Entanglement pattern: 'full', 'nearest', or 'none'.
    """

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        rotation_bases: List[str] = ["rx", "ry", "rz"],
        entanglement: str = "nearest",
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Construct a layered ansatz with explicit encoding and variational parameters.

        Returns
        -------
        circuit : QuantumCircuit
            The variational circuit.
        encoding : Iterable[ParameterVector]
            Parameter vectors that encode classical data.
        weights : Iterable[ParameterVector]
            Parameter vectors that control the variational gates.
        observables : List[SparsePauliOp]
            Pauli operators whose expectation values are used as logits.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth * len(rotation_bases))

        circuit = QuantumCircuit(num_qubits)

        # Data encoding: single‑parameter rotations on each qubit
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)

        weight_idx = 0
        for _ in range(depth):
            # Apply rotations
            for qubit in range(num_qubits):
                for gate in rotation_bases:
                    getattr(circuit, gate)(weights[weight_idx], qubit)
                    weight_idx += 1

            # Entanglement
            if entanglement == "full":
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        circuit.cz(i, j)
            elif entanglement == "nearest":
                for qubit in range(num_qubits - 1):
                    circuit.cz(qubit, qubit + 1)
            # 'none' implies no entanglement

        # Observables: Z on each qubit as logits
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        return circuit, list(encoding), list(weights), observables

    @staticmethod
    def evaluate_expectation(
        circuit: QuantumCircuit,
        encoding: Iterable[ParameterVector],
        weights: Iterable[ParameterVector],
        observables: List[SparsePauliOp],
        data: List[float],
    ) -> List[float]:
        """
        Execute the circuit on a state‑vector simulator and return expectation values.

        Parameters
        ----------
        circuit : QuantumCircuit
            The variational circuit.
        encoding : Iterable[ParameterVector]
            Encoding parameters.
        weights : Iterable[ParameterVector]
            Variational parameters.
        observables : List[SparsePauliOp]
            Observables to measure.
        data : List[float]
            Classical data to feed into the encoding.

        Returns
        -------
        expectations : List[float]
            Expectation values of each observable.
        """
        # Bind parameters
        param_dict = {str(p): val for p, val in zip(encoding, data)}
        bound_circuit = circuit.bind_parameters(param_dict)

        # Simulate
        backend = AerSimulator(method="statevector")
        job = backend.run(bound_circuit, shots=1024)
        result = job.result()
        statevector = result.get_statevector(bound_circuit)

        expectations = []
        for obs in observables:
            expectations.append(obs.expectation_value(statevector).real)

        return expectations


__all__ = ["QuantumClassifierModel"]
