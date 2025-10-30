"""Hybrid estimator that evaluates a Qiskit QCNN QNN and supports shot sampling.

This module builds on the Qiskit FastBaseEstimator and the QCNN QNN from the
QML reference, providing a unified interface that can evaluate multiple
observables, optionally perform stochastic measurement sampling, and
bind parameters to the variational circuit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.primitives import Estimator as StateEstimator

# Import the QCNN QNN factory from the QML seed
from QCNN import QCNN  # assumes QCNN.py is in the same package


class FastHybridEstimator:
    """Quantum hybrid estimator that wraps a QCNN QNN and can return expectation values with shot noise.

    The estimator accepts an EstimatorQNN instance (defaulting to the QCNN QNN)
    and evaluates it for a list of observables and parameter sets.  When
    `shots` is specified, it performs stochastic sampling using the
    underlying Estimator; otherwise, a deterministic state‑vector evaluation
    is used.
    """

    def __init__(self, qnn: EstimatorQNN | None = None) -> None:
        self.qnn = qnn if qnn is not None else QCNN()
        # A generic state‑vector estimator for deterministic evaluation
        self._state_estimator = StateEstimator()

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """
        Evaluate the QNN for each parameter set and observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            The quantum observables (e.g. Pauli strings) to evaluate.
        parameter_sets : Sequence[Sequence[float]]
            A sequence of weight vectors to bind to the variational circuit.
        shots : int, optional
            If provided, perform stochastic measurement sampling with the
            specified number of shots.  If None, use a deterministic
            state‑vector evaluation.
        seed : int, optional
            Random seed for reproducible shot sampling.

        Returns
        -------
        List[List[complex]]
            A 2‑D list where each row corresponds to a parameter set and each
            column corresponds to an observable expectation value.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            # Bind the weight parameters to the QCNN circuit
            bound_qnn = self.qnn.bind_parameters(params)

            if shots is None:
                # Deterministic state‑vector evaluation
                state = Statevector.from_instruction(bound_qnn)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Stochastic sampling using the underlying Estimator
                self._state_estimator.set_options(shots=shots, seed=seed)
                result = self._state_estimator.run(bound_qnn, shots=shots).result()
                row = [result.get_expectation_value(obs) for obs in observables]

            results.append(row)

        return results


__all__ = ["FastHybridEstimator"]
