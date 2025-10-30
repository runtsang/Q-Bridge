import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List, Optional

class FastHybridEstimator:
    """
    Hybrid estimator that evaluates a parametrized Qiskit circuit and
    optionally adds shot noise.  The class supports batched parameters,
    multiple observables and a convenient ``measure`` helper that
    returns a NumPy array.
    """
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        if not self._parameters:
            raise ValueError("Circuit must contain parameters")

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")
        observables = list(observables or [BaseOperator.identity(self._circuit.num_qubits)])
        results: List[List[complex]] = []

        for params in parameter_sets:
            circ = self._bind(params)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy.append([complex(rng.normal(z.real, 1 / np.sqrt(shots)),
                                  rng.normal(z.imag, 1 / np.sqrt(shots)))
                          for z in row])
        return noisy

    def measure(self, parameter_sets: Sequence[Sequence[float]]) -> np.ndarray:
        """Convenience wrapper that returns a NumPy array of expectation
        values for the default identity observable."""
        return np.array(self.evaluate(
            observables=[BaseOperator.identity(self._circuit.num_qubits)],
            parameter_sets=parameter_sets,
        ), dtype=np.complex128)
