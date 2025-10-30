from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator
from collections.abc import Iterable, Sequence
from typing import List

class QuantumFastBaseEstimator:
    """Evaluate expectation values of a parameterised QuantumCircuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class QuantumFastEstimator(QuantumFastBaseEstimator):
    """Add shot‑like noise via Gaussian sampling to expectation values."""
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        import numpy as np
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [rng.normal(complex(val).real, max(1e-6, 1 / shots)).real for val in row]
            noisy.append(noisy_row)
        return noisy

def EstimatorQNN(
    input_dim: int = 1,
    weight_dim: int = 1,
) -> QiskitEstimatorQNN:
    """
    Build a simple one‑qubit variational circuit with one input and one
    weight parameter.  The circuit is wrapped in a qiskit_machine_learning
    :class:`EstimatorQNN` which can be used as a quantum neural network.
    """
    inp = Parameter("input")
    wt = Parameter("weight")
    qc = QuantumCircuit(input_dim)
    if input_dim == 1:
        qc.h(0)
        qc.ry(inp, 0)
        qc.rx(wt, 0)
    else:
        qc.h(range(input_dim))
        for i in range(input_dim):
            qc.ry(inp, i)
        qc.rx(wt, 0)  # simple weight on first qubit

    observable = SparsePauliOp.from_list([("Y" * input_dim, 1)])
    estimator = StatevectorEstimator()
    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[inp],
        weight_params=[wt],
        estimator=estimator,
    )

__all__ = ["EstimatorQNN", "QuantumFastBaseEstimator", "QuantumFastEstimator"]
