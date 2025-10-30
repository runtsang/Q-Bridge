"""Hybrid estimator for quantum circuits with optional shot noise.

This module implements a lightweight estimator that can evaluate a
parameterised QuantumCircuit or a callable that returns a new circuit
given a set of parameters.  It mirrors the classical version but uses
qiskit Statevector or Aer simulators to compute expectation values.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Callable, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.opflow import PauliSumOp, StateFn, ExpectationFactory

class HybridEstimator:
    """
    Evaluate a parameterised quantum circuit for a set of observables.

    Parameters
    ----------
    circuit : QuantumCircuit | Callable[[Sequence[float]], QuantumCircuit]
        If *circuit* is a QuantumCircuit it is used directly.  If it is a
        callable, it must return a new circuit given a sequence of parameter
        values.
    shots : int | None, default None
        Number of shots to use when evaluating expectation values.  If
        ``None`` the state‑vector simulator is used, otherwise a Gaussian
        shot‑noise term with variance 1/shots is added to the deterministic
        expectation values.
    seed : int | None, default None
        Random seed for reproducibility of the noise term.
    """

    def __init__(
        self,
        circuit: QuantumCircuit | Callable[[Sequence[float]], QuantumCircuit],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.shots = shots
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _bind(self, params: Sequence[float]) -> QuantumCircuit:
        if isinstance(self.circuit, QuantumCircuit):
            if len(params)!= len(self.circuit.parameters):
                raise ValueError("Parameter count mismatch for bound circuit.")
            binding = dict(zip(self.circuit.parameters, params))
            return self.circuit.assign_parameters(binding, inplace=False)
        elif callable(self.circuit):
            return self.circuit(params)
        else:
            raise TypeError("Unsupported circuit type.")

    def evaluate(
        self,
        observables: Iterable[PauliSumOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            circ = self._bind(params)
            sv = Statevector.from_instruction(circ)
            row = []
            for obs in observables:
                op = StateFn(obs, is_measurement=True)
                exp_val = ExpectationFactory.build(op).convert(StateFn(sv)).eval()
                row.append(exp_val)
            if self.shots is not None:
                row = [complex(self._rng.normal(float(val.real), max(1e-6, 1 / self.shots))) for val in row]
            results.append(row)
        return results

    def evaluate_batch(self, *args, **kwargs) -> List[List[complex]]:
        return self.evaluate(*args, **kwargs)

__all__ = ["HybridEstimator"]
