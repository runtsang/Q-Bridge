"""Hybrid estimator for quantum circuits with optional shot sampling."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator


class HybridBaseEstimator:
    """Evaluate expectation values of observables for a parametrized quantum circuit, with optional shot sampling."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        if shots is None:
            return self._evaluate_deterministic(observables, parameter_sets)
        return self._evaluate_noisy(observables, parameter_sets, shots, seed)

    def _evaluate_deterministic(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results

    def _evaluate_noisy(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
        seed: int | None,
    ) -> List[List[complex]]:
        backend = Aer.get_backend("qasm_simulator")
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            job = execute(
                bound,
                backend=backend,
                shots=shots,
                seed_simulator=seed,
                seed_transpiler=seed,
            )
            counts = job.result().get_counts()
            # Simplified conversion to expectation values
            row = []
            for observable in observables:
                exp = 0.0
                for bitstring, count in counts.items():
                    # Placeholder: map bitstring to eigenvalue (+1/-1)
                    z = 1 if bitstring[-1] == "1" else -1
                    exp += z * count / shots
                row.append(exp)
            results.append(row)
        return results


def build_classifier_circuit_quantum(
    num_qubits: int, depth: int
) -> tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


__all__ = ["HybridBaseEstimator", "build_classifier_circuit_quantum"]
