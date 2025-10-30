"""FastBaseEstimator using Pennylane for quantum expectation evaluation.

Features:
- Variational circuit support via QNode.
- Shot noise via configurable device shots.
- Automatic conversion from Qiskit BaseOperator to Pennylane QubitUnitary.
- Gradient estimation using parameter shift rule for all observables.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator


def _convert_to_qml_operator(op: BaseOperator) -> qml.QubitUnitary:
    """Convert a Qiskit BaseOperator to a Pennylane QubitUnitary."""
    num_qubits = int(np.log2(op.shape[0]))
    return qml.QubitUnitary(op.data, wires=range(num_qubits))


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""

    def __init__(self, circuit: QuantumCircuit, shots: int = 1024, dev_name: str = "default.qubit"):
        self.circuit = circuit
        self.shots = shots
        self.dev = qml.device(dev_name, wires=circuit.num_qubits, shots=shots)
        self._params = list(circuit.parameters)
        self._observables: List[qml.Operator] = []

    def _qnode_func(self, *params):
        """Internal QNode function that returns expectation values."""
        param_map = dict(zip(self._params, params))
        bound_circ = self.circuit.assign_parameters(param_map, inplace=False)
        return [qml.expval(obs) for obs in self._observables]

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of Qiskit BaseOperator instances.
        parameter_sets
            Sequence of parameter lists to evaluate.

        Returns
        -------
        np.ndarray
            Shape (n_sets, n_observables) containing the results.
        """
        self._observables = [_convert_to_qml_operator(op) for op in observables]
        qnode = qml.QNode(self._qnode_func, self.dev)

        results: List[List[complex]] = []
        for params in parameter_sets:
            res = qnode(*params)
            results.append(res)

        return np.array(results, dtype=np.complex128)

    def evaluate_grad(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """
        Compute gradients of all observables with respect to parameters.

        Parameters
        ----------
        observables
            Iterable of Qiskit BaseOperator instances.
        parameter_sets
            Sequence of parameter lists to evaluate.

        Returns
        -------
        np.ndarray
            Shape (n_sets, n_observables, n_params) containing the gradients.
        """
        self._observables = [_convert_to_qml_operator(op) for op in observables]
        qnode = qml.QNode(self._qnode_func, self.dev)

        grads: List[np.ndarray] = []
        for params in parameter_sets:
            grad = qml.grad(qnode)(*params)  # parameter shift gradient
            grads.append(grad)

        return np.array(grads, dtype=np.complex128)


__all__ = ["FastBaseEstimator"]
