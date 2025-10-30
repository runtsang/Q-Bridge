from __future__ import annotations

from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator


class ClassifierCircuitFactory:
    """
    Build a variational quantum circuit for binary classification.
    The circuit consists of an RX encoding followed by layers of Ry (and optional Rz)
    rotations and CZ entanglement.  The function returns the circuit, parameter
    vectors, and a list of Pauli‑Z observables on each qubit.
    """

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        entanglement: str = "full",
        use_ryz: bool = True,
    ) -> Tuple[
        QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]
    ]:
        """
        Construct the variational circuit.

        Parameters
        ----------
        num_qubits : int
            Number of qubits (features to encode).
        depth : int
            Number of variational layers.
        entanglement : str, optional
            Pattern of CZ gates: 'full' (all pairs) or 'linear' (nearest neighbours).
        use_ryz : bool, optional
            If True apply Ry followed by Rz rotations per qubit; otherwise only Ry.

        Returns
        -------
        circuit : QuantumCircuit
            The constructed circuit.
        encoding : List[ParameterVector]
            Parameter vector for data encoding (RX).
        weights : List[ParameterVector]
            Parameter vector for trainable rotations.
        observables : List[SparsePauliOp]
            Pauli‑Z operators on each qubit.
        """
        encoding = ParameterVector("x", num_qubits)
        # Each qubit gets a rotation per layer; optionally a second Rz rotation
        rot_per_q = 2 if use_ryz else 1
        weights = ParameterVector("theta", num_qubits * depth * rot_per_q)

        circuit = QuantumCircuit(num_qubits)

        # Data encoding: RX on each qubit
        for q, param in enumerate(encoding):
            circuit.rx(param, q)

        idx = 0
        for _ in range(depth):
            # Rotations
            for q in range(num_qubits):
                circuit.ry(weights[idx], q)
                idx += 1
                if use_ryz:
                    circuit.rz(weights[idx], q)
                    idx += 1
            # Entanglement
            if entanglement == "linear":
                for q in range(num_qubits - 1):
                    circuit.cz(q, q + 1)
            else:  # full entanglement
                for q1 in range(num_qubits):
                    for q2 in range(q1 + 1, num_qubits):
                        circuit.cz(q1, q2)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        return circuit, [encoding], [weights], observables

    @staticmethod
    def compute_expectations(
        circuit: QuantumCircuit, observables: List[SparsePauliOp]
    ) -> List[float]:
        """
        Compute expectation values of the provided Pauli observables
        on the state prepared by *circuit* using a state‑vector simulator.

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit whose state is to be measured.
        observables : List[SparsePauliOp]
            Pauli operators whose expectation values are desired.

        Returns
        -------
        List[float]
            Real parts of the expectation values for each observable.
        """
        simulator = AerSimulator(method="statevector")
        job = simulator.run(circuit)
        result = job.result()
        statevector = result.get_statevector(circuit)

        return [float(op.expectation_value(statevector).real) for op in observables]


__all__ = ["ClassifierCircuitFactory"]
