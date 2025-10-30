import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List, Union

class FastBaseEstimator:
    """Quantum estimator that evaluates expectation values for a parametrized circuit.

    Supports exact state‑vector evaluation and optional Gaussian shot noise to
    emulate measurement statistics.  The ``shots`` argument controls the
    noise level; when supplied, values are perturbed according to a normal
    distribution with standard deviation 1/√shots.
    """

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._params):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self._params, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Union[BaseOperator, SparsePauliOp]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : iterable
            Quantum operators (BaseOperator or SparsePauliOp).  Strings of
            Z‑Pauli operators are converted automatically.
        parameter_sets : sequence of sequences
            Batch of parameter values, one per circuit instance.
        shots : int or None
            If provided, Gaussian noise with std = 1/√shots is added to each
            expectation value to mimic finite‑shot sampling.
        seed : int or None
            Seed for the noise generator.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        rng = np.random.default_rng(seed) if shots is not None else None

        for vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(vals))
            row = []
            for obs in observables:
                # Convert string to SparsePauliOp if necessary
                if isinstance(obs, str):
                    op = SparsePauliOp.from_label(obs)
                else:
                    op = obs
                val = state.expectation_value(op)
                if shots is not None:
                    val = rng.normal(val, max(1e-6, 1 / shots))
                row.append(val)
            results.append(row)
        return results


__all__ = ["FastBaseEstimator"]
