import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit import Aer, execute
from collections.abc import Iterable, Sequence
from typing import List

class HybridBaseEstimator:
    """Fast quantum estimator for a parametrised circuit.

    Computes exact expectation values using a Statevector.  If ``shots`` is
    provided, the estimator samples the circuit and returns a noisy
    expectation value with Gaussian variance ``1/shots``.
    """
    def __init__(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_circuit = self._bind(values)
            state = Statevector.from_instruction(bound_circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [complex(rng.normal(np.real(v), max(1e-6, 1 / shots)),
                                     rng.normal(np.imag(v), max(1e-6, 1 / shots))) for v in row]
                noisy.append(noisy_row)
            return noisy

        return results

__all__ = ["HybridBaseEstimator"]
