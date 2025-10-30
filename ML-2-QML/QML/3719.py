"""Hybrid estimator for quantum circuits and SamplerQNN.

The class can be constructed with a plain ``QuantumCircuit`` or a
``SamplerQNN`` instance. It evaluates expectation values of a list of
``BaseOperator`` observables for multiple parameter sets, optionally
adding Gaussian shot noise. The :py:meth:`sample` method draws samples
from the circuit using a ``StatevectorSampler`` or the built‑in
``SamplerQNN`` sampler.

Example::

    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit.primitives import StatevectorSampler
    from FastHybridEstimator_qml import FastHybridEstimator

    # simple QNN
    inputs = ParameterVector('x', 2)
    weights = ParameterVector('w', 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)

    estimator = FastHybridEstimator(qc)
    obs = [qiskit.quantum_info.operators.SparsePauliOp('Z0'),...]
    params = [[0.1, 0.2], [0.3, 0.4]]
    results = estimator.evaluate(obs, params, shots=1000, seed=42)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional, List, Union

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler


class FastHybridEstimator:
    """Estimates expectations and samples from a quantum circuit or a SamplerQNN.

    Parameters
    ----------
    circuit_or_qnn : QuantumCircuit | SamplerQNN
        If a ``QuantumCircuit`` is supplied the class will bind the
        parameters and evaluate the statevector.  If a ``SamplerQNN`` is
        supplied the built‑in sampler is used for both evaluation and
        sampling.
    """

    def __init__(self, circuit_or_qnn: Union[QuantumCircuit, SamplerQNN]) -> None:
        if isinstance(circuit_or_qnn, SamplerQNN):
            self._sampler_qnn = circuit_or_qnn
            self._circuit = circuit_or_qnn.circuit
            self._parameters = list(circuit_or_qnn.input_params) + list(circuit_or_qnn.weight_params)
        else:
            self._circuit = circuit_or_qnn
            self._sampler_qnn = None
            self._parameters = list(circuit_or_qnn.parameters)

        # A generic statevector sampler for pure circuits
        self._sv_sampler = StatevectorSampler()

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    # ------------------ evaluation ------------------ #
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of ``BaseOperator`` instances.
        parameter_sets
            Sequence of 1‑D sequences of parameters.
        shots
            If supplied, Gaussian shot noise is added to each expectation
            value with variance 1/shots.
        seed
            Random seed for the Gaussian noise.

        Returns
        -------
        List[List[complex]]
            Outer list over parameter sets, inner list over observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [complex(rng.normal(val.real, max(1e-6, 1 / shots))
                                     + 1j * rng.normal(val.imag, max(1e-6, 1 / shots)))
                             for val in row]
                noisy.append(noisy_row)
            return noisy

        return results

    # ------------------ sampling ------------------ #
    def sample(
        self,
        parameter_set: Sequence[float],
        n: int = 1,
        seed: Optional[int] = None,
    ) -> List[int]:
        """Draw samples from the circuit or SamplerQNN.

        Parameters
        ----------
        parameter_set
            Sequence of parameters for a single evaluation.
        n
            Number of samples to draw.
        seed
            Random seed for reproducibility.

        Returns
        -------
        List[int]
            Sample indices in the computational basis.
        """
        if self._sampler_qnn is not None:
            probs = self._sampler_qnn.sample(parameter_set, n=1000)
            probs = np.array(probs)
        else:
            state = Statevector.from_instruction(self._bind(parameter_set))
            probs = state.probabilities()
        rng = np.random.default_rng(seed)
        return rng.choice(len(probs), size=n, p=probs).tolist()

__all__ = ["FastHybridEstimator"]
