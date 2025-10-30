"""Hybrid quantum classifier utilities.

The module defines :class:`SharedClassifier` which can build a
variational circuit and evaluate it with a fast state‑vector
estimator.  It also provides a helper to construct a
EstimatorQNN instance from Qiskit, mirroring the regression
example, and supports optional shot noise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import BaseOperator, SparsePauliOp, Statevector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StateEstimator


class SharedClassifier:
    """Factory and evaluator for hybrid quantum models."""

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int, depth: int
    ) -> Tuple[QuantumCircuit, Iterable[Parameter], Iterable[Parameter], List[SparsePauliOp]]:
        """
        Return a variational circuit with data‑encoding and a layered
        ansatz.  Parameters are split into data‑encoding ``x`` and
        variational ``theta`` groups.  The output includes a list of
        Pauli‑Z observables for each qubit.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        qc = QuantumCircuit(num_qubits)
        for param, q in zip(encoding, range(num_qubits)):
            qc.rx(param, q)

        idx = 0
        for _ in range(depth):
            for q in range(num_qubits):
                qc.ry(weights[idx], q)
                idx += 1
            for q in range(num_qubits - 1):
                qc.cz(q, q + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return qc, list(encoding), list(weights), observables

    @staticmethod
    def estimator_qnn(
        circuit: QuantumCircuit,
        observables: Iterable[BaseOperator],
        input_params: List[Parameter],
        weight_params: List[Parameter],
        estimator: StateEstimator | None = None,
    ) -> EstimatorQNN:
        """
        Wrap a circuit in Qiskit’s EstimatorQNN for regression.
        """
        if estimator is None:
            estimator = StateEstimator()
        return EstimatorQNN(
            circuit=circuit,
            observables=observables,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
        )

    @staticmethod
    def evaluate(
        circuit: QuantumCircuit,
        parameter_sets: Sequence[Sequence[float]],
        observables: Iterable[BaseOperator],
    ) -> List[List[complex]]:
        """
        Fast expectation‑value evaluation using Statevector.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = dict(zip(circuit.parameters, values))
            state = Statevector.from_instruction(circuit.assign_parameters(bound))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    @staticmethod
    def evaluate_with_shots(
        circuit: QuantumCircuit,
        parameter_sets: Sequence[Sequence[float]],
        observables: Iterable[BaseOperator],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Add Gaussian shot noise to the deterministic expectation
        values, emulating a noisy quantum backend.
        """
        raw = SharedClassifier.evaluate(circuit, parameter_sets, observables)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                complex(
                    rng.normal(val.real, 1 / shots),
                    rng.normal(val.imag, 1 / shots),
                )
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["SharedClassifier"]
