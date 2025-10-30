"""Quantum circuit factory with enriched ansatz and measurement support."""

from __future__ import annotations

from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator


class QuantumClassifierModel:
    """Factory for a quantum classifier circuit with configurable entanglement."""

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        entanglement: str = "CZ",
        measurement: str = "Z",
        verbose: bool = False,
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Construct a layered ansatz with optional entanglement and measurement patterns.

        Parameters
        ----------
        num_qubits: int
            Number of qubits / input features.
        depth: int
            Number of variational layers.
        entanglement: str
            Type of two‑qubit gate used for entanglement ("CZ" or "CNOT").
        measurement: str
            Pauli string to measure per qubit ("Z" or "X").
        verbose: bool
            If true, prints circuit summary.

        Returns
        -------
        circuit: QuantumCircuit
            The constructed circuit.
        encoding: List[Parameter]
            Parameters for data encoding.
        weights: List[Parameter]
            Variational parameters.
        observables: List[SparsePauliOp]
            Pauli operators whose expectation values are returned.
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
            # Single‑qubit rotations
            for qubit in range(num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            # Entanglement
            if entanglement.upper() == "CZ":
                for qubit in range(num_qubits - 1):
                    circuit.cz(qubit, qubit + 1)
            elif entanglement.upper() == "CNOT":
                for qubit in range(num_qubits - 1):
                    circuit.cx(qubit, qubit + 1)
            else:
                raise ValueError(f"Unsupported entanglement type: {entanglement}")

        # Measurement observables
        if measurement.upper() == "Z":
            observables = [
                SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                for i in range(num_qubits)
            ]
        elif measurement.upper() == "X":
            observables = [
                SparsePauliOp("I" * i + "X" + "I" * (num_qubits - i - 1))
                for i in range(num_qubits)
            ]
        else:
            raise ValueError(f"Unsupported measurement type: {measurement}")

        if verbose:
            print(circuit.draw(output="text"))

        return circuit, list(encoding), list(weights), observables

    @staticmethod
    def expectation_values(
        circuit: QuantumCircuit,
        params: dict,
        observables: List[SparsePauliOp],
        shots: int = 1024,
    ) -> List[float]:
        """
        Evaluate expectation values of the given observables on the circuit.

        Parameters
        ----------
        circuit: QuantumCircuit
            The circuit to execute.
        params: dict
            Mapping of Parameter to float values.
        observables: List[SparsePauliOp]
            Observables to evaluate.
        shots: int
            Number of measurement shots.

        Returns
        -------
        List[float]
            Expectation values in the same order as observables.
        """
        bound_circ = circuit.bind_parameters(params)
        simulator = AerSimulator()
        job = simulator.run(bound_circ, shots=shots, memory=True)
        result = job.result()
        counts = result.get_counts(bound_circ)

        # Convert counts to expectation values
        exp_vals = []
        for op in observables:
            exp = 0.0
            for bitstring, freq in counts.items():
                parity = 1
                for qubit, pauli in enumerate(op.to_label()):
                    if pauli == "Z" and bitstring[::-1][qubit] == "1":
                        parity *= -1
                exp += parity * freq / shots
            exp_vals.append(exp)
        return exp_vals


__all__ = ["QuantumClassifierModel"]
