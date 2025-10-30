from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class HybridEstimator:
    """
    Hybrid estimator that evaluates parameterised quantum circuits with
    optional shot noise.

    Parameters
    ----------
    circuit
        A qiskit QuantumCircuit instance.
    backend
        Backend for execution; defaults to Aer.get_backend('qasm_simulator').

    The ``evaluate`` method accepts any iterable of qiskit BaseOperator
    instances as observables.  For shot‑based execution the routine
    transforms measurement counts into expectation values and can add
    Gaussian noise to mimic classical measurement uncertainty.
    """

    def __init__(self, circuit: QuantumCircuit, backend=None) -> None:
        self.circuit = circuit
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute expectation values for each observable and parameter set.

        Parameters
        ----------
        observables
            Iterable of qiskit BaseOperator instances.
        parameter_sets
            Sequence of parameter vectors.
        shots
            Number of shots to simulate; if None, uses state‑vector evaluation.
        seed
            Random seed for shot‑noise simulation.

        Returns
        -------
        List[List[float]]
            Outer list per parameter set, inner list per observable.
        """
        observables = list(observables)
        results: List[List[float]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            if shots is None:
                state = Statevector.from_instruction(bound)
                row = [float(op.expectation_value(state)) for op in observables]
            else:
                job = execute(bound, self.backend, shots=shots)
                result = job.result()
                counts = result.get_counts(bound)
                probs = {k: v / shots for k, v in counts.items()}
                row = []
                for op in observables:
                    exp = 0.0
                    for bitstring, p in probs.items():
                        eig = int(bitstring, 2)
                        exp += eig * p
                    row.append(exp)

            if shots is not None and seed is not None:
                rng = np.random.default_rng(seed)
                row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            results.append(row)

        return results


__all__ = ["HybridEstimator"]
