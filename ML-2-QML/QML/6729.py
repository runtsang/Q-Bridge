"""Quantum estimator based on Pennylane variational circuits."""
from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from typing import Iterable, List, Callable

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized Pennylane circuit."""
    def __init__(self, num_qubits: int, circuit: Callable):
        """
        Parameters
        ----------
        num_qubits : int
            Number of qubits in the circuit.
        circuit : Callable
            A function that applies gates to the default device and accepts *params
            as its arguments.
        """
        self.num_qubits = num_qubits
        self.circuit = circuit
        self.device = qml.device("default.qubit", wires=num_qubits)

    def _make_qnode(self, observables: Iterable[qml.operation.Operator]) -> qml.QNode:
        @qml.qnode(self.device, interface="numpy")
        def qnode(*params):
            self.circuit(*params)
            return [qml.expval(obs) for obs in observables]
        return qnode

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: List[List[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.
        """
        qnode = self._make_qnode(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            results.append(qnode(*params))
        return results

__all__ = ["FastBaseEstimator"]
