"""Hybrid fast estimator for Qiskit circuits with optional shot noise.

This module mirrors the classical implementation but operates on
parameter‑ised quantum circuits.  It also exposes a quantum
classifier circuit builder that follows the same API as the classical
variant.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Iterable as IterableType, List, Tuple

from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators import SparsePauliOp

ScalarObservable = BaseOperator  # for type clarity


class FastBaseEstimator:
    """Evaluate expectation values of a parametrised circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to evaluate.  It must be fully parameterised.
    """

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
        observables: Iterable[BaseOperator] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Return a matrix of expectation values.

        Each row corresponds to a parameter set and each column to an
        observable.  With ``shots`` a state‑vector simulation is replaced
        by a shot‑based Aer simulator to emulate measurement noise.
        """
        if observables is None:
            observables = []
        if parameter_sets is None:
            return []

        observables = list(observables)
        results: List[List[complex]] = []

        backend = Aer.get_backend("statevector_simulator")
        if shots is not None:
            backend = Aer.get_backend("qasm_simulator")

        for values in parameter_sets:
            bound = self._bind(values)

            if shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(
                    bound,
                    backend=backend,
                    shots=shots,
                    seed_simulator=seed,
                    seed_transpiler=seed,
                )
                result = job.result()
                counts = result.get_counts()
                # convert counts to expectation values
                row = [
                    _pauli_expectation(counts, obs, bound.num_qubits)
                    for obs in observables
                ]

            results.append(row)

        return results


def _pauli_expectation(counts: dict[str, int], op: BaseOperator, n_qubits: int) -> complex:
    """Compute expectation value of a Pauli operator from shot counts."""
    total = sum(counts.values())
    exp_val = 0.0
    for bitstring, count in counts.items():
        # Qiskit bitstring order is little‑endian
        bs = bitstring[::-1]
        val = 1.0
        for idx, bit in enumerate(bs):
            if op.paulis[idx] == "Z" and bit == "1":
                val *= -1
        exp_val += val * count
    return exp_val / total


# --------------------------------------------------------------------------- #
#  Classifier circuit builder – quantum analogue of the classical helper
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Return a simple variational ansatz with metadata.

    The returned tuple mimics the signature of the classical helper:
    ``(circuit, encoding, weights, observables)``.
    """
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


__all__ = ["FastBaseEstimator", "build_classifier_circuit"]
