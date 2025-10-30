"""Hybrid fully‑connected quantum layer with parameterized circuit and observable support.

The class mirrors the classical counterpart but uses Qiskit for circuit construction,
state‑vector simulation, and optional shot‑based sampling.  It exposes a ``run`` method
for single‑parameter inference and an ``evaluate`` method for batch evaluation of arbitrary
Pauli/Z observables (FastBaseEstimator style).
"""

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence

class FCLHybridQuantum:
    """
    Parameterized quantum circuit that emulates a fully‑connected layer.
    Supports:
    * single‑parameter inference with optional shot noise
    * batch evaluation of arbitrary observable operators
    """

    def __init__(self, n_qubits: int = 1, shots: int | None = None) -> None:
        self._circuit = QuantumCircuit(n_qubits)
        self._theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.ry(self._theta, range(n_qubits))
        self._circuit.measure_all()
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

    def _bind(self, theta: float) -> QuantumCircuit:
        """Return a new circuit with the parameter bound."""
        return self._circuit.assign_parameters({self._theta: theta}, inplace=False)

    def run(self, theta: float) -> np.ndarray:
        """Return the expectation value for a single parameter."""
        if self.shots is None:
            # Deterministic state‑vector evaluation
            state = Statevector.from_instruction(self._bind(theta))
            expectation = state.expectation_value(qiskit.quantum_info.operators.Pauli("Z"))
            return np.array([float(expectation)])
        # Shot‑based simulation
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self._theta: theta}],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array([int(k, 2) for k in result.keys()])
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Batch evaluation of multiple parameter sets and observables.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Qiskit operators (e.g., Pauli, Identity) to measure.
        parameter_sets : Sequence[Sequence[float]]
            Each inner sequence contains the parameters for one run.

        Returns
        -------
        List[List[complex]]
            Matrix of expectation values for every parameter set and observable.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            theta = params[0]
            state = Statevector.from_instruction(self._bind(theta))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


def FCL() -> FCLHybridQuantum:
    """Convenience factory matching the original FCL anchor."""
    return FCLHybridQuantum()
