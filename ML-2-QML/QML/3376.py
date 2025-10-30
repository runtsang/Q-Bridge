"""
Quantum implementation of a hybrid fully‑connected layer estimator.
Uses a parameterised circuit with Ry rotations and provides a
Fast‑style batched evaluation interface.
"""

from __future__ import annotations

from collections.abc import Iterable as IterableType, Sequence
from typing import List, Optional

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FCLHybridEstimator:
    """
    Quantum surrogate for a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits / input parameters.
    backend : qiskit.providers.backend.Backend, optional
        Backend to execute the circuit.  Defaults to Aer qasm_simulator.
    shots : int, optional
        Number of shots for expectation estimation.  If ``None`` the
        state‑vector simulator is used.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        backend: Optional[qiskit.providers.backend.Backend] = None,
        shots: Optional[int] = 100,
    ) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        self._circuit = QuantumCircuit(n_qubits)
        self.theta = Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

    def _bind(self, parameters: Sequence[float]) -> QuantumCircuit:
        if len(parameters)!= self.n_qubits:
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = {self.theta: parameters[0]}
        return self._circuit.assign_parameters(mapping, inplace=False)

    def run(self, thetas: IterableType[float]) -> np.ndarray:
        """Execute the circuit for a single set of parameters."""
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

    def evaluate(
        self,
        observables: IterableType[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """
        Evaluate a batch of observables for each parameter set.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Pauli or other operators to measure expectation values of.
        parameter_sets : sequence of sequences
            Batches of parameters to evaluate.
        shots : int, optional
            Override the instance shots for this call.
        seed : int, optional
            Random seed for reproducible shot noise (only used when shots is None).
        """
        obs_list = list(observables)
        if not obs_list:
            return []

        results: List[List[complex]] = []
        effective_shots = shots if shots is not None else self.shots

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in obs_list]
            results.append(row)

        if shots is None and seed is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[complex]] = []
            for row in results:
                noisy_row = [
                    rng.normal(c.real, max(1e-6, 1 / effective_shots))
                    + 1j * rng.normal(c.imag, max(1e-6, 1 / effective_shots))
                    for c in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results


def FCL(n_qubits: int = 1, shots: int = 100) -> FCLHybridEstimator:
    return FCLHybridEstimator(n_qubits, None, shots)


__all__ = ["FCLHybridEstimator", "FCL"]
