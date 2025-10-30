"""Quantum‑centric FastBaseEstimator built on PennyLane.

Features
--------
* Parameterized circuit evaluation on any PennyLane device.
* Supports deterministic state‑vector or shot‑based simulators.
* Vectorised evaluation of multiple observables per parameter set.
* Optional gradient computation via parameter‑shift rule.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence, Union

import pennylane as qml
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to evaluate. Parameters are bound at runtime.
    device : str | qml.Device, optional
        PennyLane device to use. Defaults to ``"default.qubit"``.
    """

    def __init__(self, circuit: QuantumCircuit, device: str | qml.Device = "default.qubit") -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)
        self._device = qml.device(device, wires=circuit.num_qubits)

    def _qnode_factory(self, observable, shots: int | None = None):
        """Create a PennyLane QNode that returns the expectation of ``observable``."""
        if isinstance(observable, BaseOperator):
            pl_obs = qml.from_qiskit(observable)
        else:
            pl_obs = observable

        @qml.qnode(self._device, shots=shots, interface="jax")
        def circuit_fn(*params):
            bound = self._circuit.assign_parameters(dict(zip(self._params, params)), inplace=False)
            pl_circ = qml.from_qiskit(bound, wires=range(bound.num_qubits))
            return qml.expval(pl_obs)(pl_circ)

        return circuit_fn

    def evaluate(
        self,
        observables: Iterable[Union[BaseOperator, qml.operation.Operator]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables
            List of qiskit BaseOperator or PennyLane Operator.
        parameter_sets
            Sequence of parameter vectors.
        shots
            Number of shots for a shot‑based device. If ``None`` the device
            is used in state‑vector mode.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            if shots is None:
                state = Statevector.from_instruction(
                    self._circuit.assign_parameters(dict(zip(self._params, values)), inplace=False)
                )
                row = [state.expectation_value(obs) for obs in observables]
            else:
                row = []
                for obs in observables:
                    qnode = self._qnode_factory(obs, shots)
                    row.append(qnode(*values))
            results.append(row)
        return results

    def evaluate_with_gradients(
        self,
        observable: Union[BaseOperator, qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[tuple[complex, np.ndarray]]:
        """Return expectation values and gradients for a single observable.

        The gradient is computed using the parameter‑shift rule.
        """
        results: List[tuple[complex, np.ndarray]] = []

        for values in parameter_sets:
            if shots is None:
                state = Statevector.from_instruction(
                    self._circuit.assign_parameters(dict(zip(self._params, values)), inplace=False)
                )
                exp_val = state.expectation_value(observable)
                grad = np.zeros(len(values))
                for i in range(len(values)):
                    shift = np.pi / 2
                    plus = values.copy()
                    minus = values.copy()
                    plus[i] += shift
                    minus[i] -= shift
                    plus_state = Statevector.from_instruction(
                        self._circuit.assign_parameters(dict(zip(self._params, plus)), inplace=False)
                    )
                    minus_state = Statevector.from_instruction(
                        self._circuit.assign_parameters(dict(zip(self._params, minus)), inplace=False)
                    )
                    grad[i] = (plus_state.expectation_value(observable) - minus_state.expectation_value(observable)) / 2
            else:
                qnode = self._qnode_factory(observable, shots)
                exp_val = qnode(*values)
                grad = np.array(qnode.gradients(*values))
            results.append((exp_val, grad))
        return results


__all__ = ["FastBaseEstimator"]
