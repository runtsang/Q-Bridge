"""FastHybridEstimator: hybrid quantum estimator for Qiskit circuits with optional shot noise.

The class accepts a parametrized QuantumCircuit and evaluates expectation values of a
set of BaseOperator observables for many parameter configurations.  An AerSimulator
with the ``statevector`` backend is used for deterministic evaluation; when a
shot count is supplied, the simulator runs with the specified shots, producing
stochastic estimates that emulate real device measurement statistics.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Statevector

# Backends
_statevector_sim = Aer.get_backend("statevector_simulator")
_qasm_sim = Aer.get_backend("qasm_simulator")


class FastHybridEstimator:
    """Evaluate a parametrized quantum circuit for multiple parameter sets.

    Parameters
    ----------
    circuit
        A QuantumCircuit with symbolic parameters.
    shots
        When provided, the simulator runs with ``shots`` to produce noisy
        expectation values.  If ``None`` the statevector backend is used.
    seed
        Random seed forwarded to the simulator for reproducibility.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.shots = shots
        self.seed = seed

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return a list of rows, one per parameter set, each containing
        the expectation values of all observables."""
        observables = list(observables)
        results: List[List[complex]] = []

        if self.shots is None:
            # Deterministic state‑vector evaluation
            for values in parameter_sets:
                circuit = self._bind(values)
                state = Statevector.from_instruction(circuit)
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
        else:
            # Stochastic evaluation with shots
            job_kwargs = {"shots": self.shots}
            if self.seed is not None:
                job_kwargs["seed_simulator"] = self.seed
            for values in parameter_sets:
                circuit = self._bind(values)
                job = execute(circuit, _qasm_sim, **job_kwargs)
                result = job.result()
                counts = result.get_counts(circuit)
                # Build a weighted statevector from the measurement counts
                state = Statevector.from_counts(circuit, counts)
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
        return results


__all__ = ["FastHybridEstimator"]
