"""Extended FastBaseEstimator for Pennylane quantum circuits with shot‑noise support."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Callable

import pennylane as qml
from pennylane.operation import Operation


class FastBaseEstimator:
    """Evaluate expectation values of Pennylane observables for a parametrized circuit."""
    def __init__(
        self,
        circuit_fn: Callable[..., None],
        wires: int,
        device_name: str = "default.qubit",
    ) -> None:
        """
        Parameters
        ----------
        circuit_fn
            Function that applies gates to the Pennylane device. It must accept the same
            number of positional arguments as there are parameters.
        wires
            Number of qubits in the circuit.
        device_name
            Pennylane backend to use (e.g. 'default.qubit', 'qiskit.ibmq_qasm_simulator').
        """
        self.device = qml.device(device_name, wires=wires)
        self.circuit_fn = circuit_fn
        self.observables: List[Operation] = []

        @qml.qnode(self.device, interface="torch")
        def _qnode(*params: float):
            self.circuit_fn(*params)
            return [qml.expval(obs) for obs in self.observables]

        self._qnode = _qnode

    def evaluate(
        self,
        observables: Iterable[Operation],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute expectation values for each parameter set.

        Parameters
        ----------
        observables
            Iterable of Pennylane operations whose expectation values are to be measured.
        parameter_sets
            Sequence of parameter tuples.
        shots
            If provided, the circuit is executed with this number of shots, adding
            sampling noise.  Otherwise the exact expectation value is returned.
        seed
            Random seed for the shot‑noise simulation.
        """
        self.observables = list(observables)
        results: List[List[float]] = []
        for params in parameter_sets:
            if shots is None:
                vals = self._qnode(*params)
            else:
                vals = self._qnode(*params, shots=shots, seed=seed)
            results.append([float(v) for v in vals])
        return results


__all__ = ["FastBaseEstimator"]
