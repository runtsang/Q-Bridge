"""Quantum classifier builder and evaluator using Qiskit."""

from __future__ import annotations

from typing import Iterable, Sequence, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.providers.aer import AerSimulator


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    data_encoding: str = "rx",
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], List[int], List[SparsePauliOp]]:
    """
    Construct a layered data‑re‑uploading Ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits, equal to the number of features.
    depth : int
        Number of variational layers.
    data_encoding : str
        Gate used for data encoding ('rx', 'ry', or 'rz').

    Returns
    -------
    circuit : QuantumCircuit
        Parameterised circuit.
    encoding : Iterable[ParameterVector]
        Parameter vectors used for data encoding.
    weight_sizes : List[int]
        Number of variational parameters per layer.
    observables : List[SparsePauliOp]
        Default Pauli observables for expectation measurement.
    """
    # Data‑encoding parameters
    encoding = ParameterVector("x", num_qubits)
    # Variational parameters
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for idx, qubit in enumerate(range(num_qubits)):
        if data_encoding == "rx":
            circuit.rx(encoding[idx], qubit)
        elif data_encoding == "ry":
            circuit.ry(encoding[idx], qubit)
        else:
            circuit.rz(encoding[idx], qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: local Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    # Count parameters per layer
    weight_sizes = [num_qubits] * depth

    return circuit, encoding, weight_sizes, observables


class QuantumClassifierModel:
    """
    Quantum classifier wrapper exposing the same interface as its classical twin.

    The class holds a parameterised circuit and can evaluate a list of
    Pauli observables for a batch of parameter sets.  Optional shot noise
    emulates measurement statistics.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        backend: str = "aer_simulator",
    ) -> None:
        self.circuit, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.backend = backend
        self.simulator = AerSimulator()

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : Iterable[SparsePauliOp] | None
            If None, default observables from ``build_classifier_circuit`` are used.
        parameter_sets : Sequence[Sequence[float]] | None
            Batch of parameter vectors (data + variational parameters).
        shots : int | None
            Optional shot number for Gaussian noise injection.
        seed : int | None
            Random seed for reproducible noise.
        """
        if parameter_sets is None:
            return []

        obs = list(observables) or self.observables
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_circ = self._bind(values)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(observable) for observable in obs]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy_results: List[List[complex]] = []
            for row in results:
                noisy_row = [
                    complex(
                        rng.normal(val.real, max(1e-6, 1 / shots)),
                        rng.normal(val.imag, max(1e-6, 1 / shots)),
                    )
                    for val in row
                ]
                noisy_results.append(noisy_row)
            return noisy_results

        return results


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
