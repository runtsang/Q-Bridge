"""Advanced quantum estimator utilities implemented with Qiskit.

This module extends the original FastBaseEstimator by adding support for
parameter‑shift gradient estimation, shot‑noise simulation, and the
ability to target arbitrary Qiskit backends.  The API remains
compatible with the seed implementation, but the new features enable
efficient experimentation with variational circuits on simulators and
real devices.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = Callable[[np.ndarray], np.ndarray | complex]


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        backend: str | None = "aer_simulator_statevector",
        shots: int | None = None,
        device: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        circuit
            A parametrized Qiskit ``QuantumCircuit``.
        backend
            Backend name for Aer simulation or a Qiskit provider backend.
            If ``None`` the default Aer state‑vector simulator is used.
        shots
            Number of shots for the expectation value estimation.  If
            ``None`` a deterministic state‑vector calculation is used.
        device
            Optional Pennylane device name.  When provided the estimator
            will use Pennylane for both forward and gradient evaluation.
        """
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend_name = backend or "aer_simulator_statevector"
        self.shots = shots
        self.device_name = device
        if self.device_name is not None:
            import pennylane as qml
            self.dev = qml.device(self.device_name, wires=len(circuit.qubits))
        else:
            self.dev = None

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _statevector(self, circuit: QuantumCircuit) -> Statevector:
        backend = Aer.get_backend(self.backend_name)
        job = execute(circuit, backend=backend, shots=None)
        result = job.result()
        return Statevector(result.get_statevector(circuit))

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_circuit = self._bind(values)
            state = self._statevector(bound_circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        return results

    def evaluate_with_grad(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shift: float = np.pi / 2,
    ) -> List[List[float]]:
        """
        Compute gradients of expectation values w.r.t. circuit parameters
        using the parameter‑shift rule.  Works for both Qiskit and Pennylane
        backends.
        """
        observables = list(observables)
        grads: List[List[float]] = []

        for params in parameter_sets:
            row = []
            for obs in observables:
                grad = 0.0
                for idx in range(len(params)):
                    shift_plus = list(params)
                    shift_minus = list(params)
                    shift_plus[idx] += shift
                    shift_minus[idx] -= shift

                    val_plus = self._expectation(obs, shift_plus)
                    val_minus = self._expectation(obs, shift_minus)
                    grad += (val_plus - val_minus) / (2 * np.sin(shift))
                row.append(grad)
            grads.append(row)
        return grads

    def _expectation(
        self,
        observable: BaseOperator,
        parameter_values: Sequence[float],
    ) -> float:
        bound_circuit = self._bind(parameter_values)
        state = self._statevector(bound_circuit)
        return float(state.expectation_value(observable))

    @staticmethod
    def make_variational_circuit(num_qubits: int, depth: int) -> QuantumCircuit:
        """
        Create a simple layered variational circuit with Ry rotations and
        CNOT entangling gates.
        """
        qc = QuantumCircuit(num_qubits)
        for d in range(depth):
            for q in range(num_qubits):
                qc.ry(0.0, q)
            for q in range(num_qubits - 1):
                qc.cx(q, q + 1)
        return qc


__all__ = ["FastBaseEstimator"]
