"""Hybrid estimator combining quantum circuit evaluation and sampling.

This module defines :class:`FastHybridEstimator` which can evaluate a
parameterised Qiskit circuit for a list of observables, optionally add
shot‑noise, and provide a unified sampling interface via a quantum
``SamplerQNN``.  The implementation mirrors the original
``FastBaseEstimator`` but extends it with a quantum sampler and a
parameterised circuit that can be reused for both expectation‑value
and sampling tasks.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, BaseOperator
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler


class FastHybridEstimator:
    """Evaluate a Qiskit circuit for a set of parameters and observables.

    Parameters
    ----------
    circuit : QuantumCircuit
        The parameterised circuit to evaluate.  All parameters must be
        bound before execution.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
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
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        If ``shots`` is provided, the expectation values are estimated
        from a finite number of state‑vector samples, adding realistic
        shot‑noise.  When ``shots`` is ``None`` the exact state‑vector
        expectation is returned.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if shots is None:
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
            return results

        # Shot‑noise evaluation using the state‑vector sampler
        sampler = Sampler()
        for values in parameter_sets:
            bound = self._bind(values)
            shots_res = sampler.run(bound, shots=shots, seed=seed).result()
            probs = shots_res.get_counts()
            # Convert counts to expectation values
            row: List[complex] = []
            for obs in observables:
                exp = 0.0
                for bitstring, count in probs.items():
                    # Interpret bitstring as computational basis state
                    state = Statevector.from_label(bitstring)
                    exp += count * state.expectation_value(obs)
                exp /= shots
                row.append(exp)
            results.append(row)
        return results

    def sample(
        self,
        sampler: QSamplerQNN,
        parameter_sets: Sequence[Sequence[float]],
        num_samples: int,
        *,
        seed: Optional[int] = None,
    ) -> List[List[int]]:
        """Draw samples from the quantum sampler.

        Parameters
        ----------
        sampler : QSamplerQNN
            A Qiskit neural‑network sampler that implements a
            parameterised quantum circuit with a ``sample`` method.
        parameter_sets : Sequence[Sequence[float]]
            Each inner sequence contains the 2‑dimensional input for the
            sampler.
        num_samples : int
            Number of samples to draw per input.
        seed : int, optional
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        samples: List[List[int]] = []

        for params in parameter_sets:
            bound_circ = sampler.circuit.assign_parameters(
                dict(zip(sampler.input_params, params)), inplace=False
            )
            # Use the built‑in sampler primitive to draw samples
            sampler_primitive = Sampler()
            result = sampler_primitive.run(bound_circ, shots=num_samples, seed=seed).result()
            counts = result.get_counts()
            # Flatten counts into a list of outcomes
            outcome = []
            for bitstring, count in counts.items():
                outcome.extend([int(bit) for bit in bitstring] * count)
            samples.append(outcome)
        return samples


def create_qsampler_qnn() -> QSamplerQNN:
    """Return a parameterised quantum sampler with 2‑dimensional input
    and 4‑dimensional weight parameters.
    """
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    sampler = Sampler()
    sampler_qnn = QSamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)
    return sampler_qnn


__all__ = ["FastHybridEstimator", "create_qsampler_qnn"]
