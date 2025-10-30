from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class HybridFastEstimator:
    """Hybrid estimator that can evaluate either a QuantumCircuit or a PyTorch model.

    The interface mirrors the classical implementation: ``evaluate`` receives
    a list of observables and a series of parameter sets.  When a circuit is
    provided, expectation values are computed with a :class:`Statevector`;
    when a callable model is given, the call is forwarded to the model and
    the observables are applied to its output.
    """

    def __init__(self, item: Union[QuantumCircuit, Callable[[Sequence[float]], Sequence[float]]]) -> None:
        self.item = item
        if isinstance(item, QuantumCircuit):
            self._params = list(item.parameters)
        else:
            self._params = None

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, values))
        return self.item.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Union[BaseOperator, Callable[[Sequence[float]], float]]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables
            For quantum mode: BaseOperator instances.  For model mode: callables
            that map a model output to a scalar.
        parameter_sets
            Parameter values to evaluate.
        shots
            If supplied, Gaussian shot noise is added to each mean value.
        seed
            Random seed for reproducibility.
        """
        results: List[List[complex]] = []

        if isinstance(self.item, QuantumCircuit):
            for params in parameter_sets:
                state = Statevector.from_instruction(self._bind(params))
                row = [state.expectation_value(obs) for obs in observables]  # type: ignore
                results.append(row)
        else:
            # Assume callable model that returns a sequence of floats
            for params in parameter_sets:
                output = self.item(params)
                row = [obs(output) for obs in observables]  # type: ignore
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [complex(rng.normal(float(val), max(1e-6, 1 / shots))) for val in row]
            noisy.append(noisy_row)
        return noisy

class SamplerQNN:
    """Quantum version of the lightweight sampler network.

    This implements the same functional form as the classical
    :class:`SamplerQNN` but uses a parameterised two‑qubit circuit
    and a :class:`StatevectorSampler` backend.
    """

    def __init__(self) -> None:
        from qiskit.circuit import ParameterVector
        from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
        from qiskit.primitives import StatevectorSampler

        inputs = ParameterVector("x", 2)
        weights = ParameterVector("w", 4)

        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)

        sampler = StatevectorSampler()
        self.qnn = QSamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler,
            interpret=lambda x: x,
            output_shape=2,
        )

    def __call__(self, *args, **kwargs):
        return self.qnn(*args, **kwargs)

__all__ = [
    "HybridFastEstimator",
    "SamplerQNN",
]
