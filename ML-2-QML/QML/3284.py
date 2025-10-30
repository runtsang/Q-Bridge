"""Combined quantum estimator with optional sampling noise and sampler network.

This module defines FastBaseEstimatorGen224 that can evaluate a
parameterized QuantumCircuit for multiple parameter sets and
observables.  It also exposes a SamplerQNN class that builds a
simple variational sampler using Qiskit.  The design merges the
classical and quantum seeds, adding shot‑noise simulation and a
unified interface.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class SamplerQNN:
    """Variational sampler network constructed with Qiskit.

    Mirrors the structure of the classical SamplerQNN but uses a
    parameterized circuit.  The circuit is a simple 2‑qubit entangling
    block with Ry rotations for input and weight parameters.
    """
    def __init__(self) -> None:
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

        self.circuit = qc
        self.input_params = inputs
        self.weight_params = weights

    def bind(self, inputs: Sequence[float], weights: Sequence[float]) -> QuantumCircuit:
        """Bind numerical values to the circuit parameters."""
        mapping = dict(zip(self.input_params, inputs))
        mapping.update(dict(zip(self.weight_params, weights)))
        return self.circuit.assign_parameters(mapping, inplace=False)


class FastBaseEstimatorGen224:
    """Evaluate a Qiskit circuit for batches of parameters and observables.

    Parameters
    ----------
    circuit : QuantumCircuit
        The parameterized circuit to evaluate.
    noise_std : float | None, optional
        Standard deviation of Gaussian noise added to sampled expectation
        values.  Mimics shot noise when a sampler is used.
    seed : int | None, optional
        Random seed for reproducibility of the noise.
    """
    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        noise_std: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.circuit = circuit
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Operators whose expectation values are required.
        parameter_sets : Sequence[Sequence[float]]
            Each entry is a list of parameters matching the circuit's
            parameter vector.

        Returns
        -------
        List[List[complex]]
            Nested list of expectation values per parameter set.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            bound_circ = self.circuit.assign_parameters(
                dict(zip(self.circuit.parameters, params)), inplace=False
            )
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            if self.noise_std is not None:
                noise = self.rng.normal(0, self.noise_std, size=len(row))
                row = [val + noise[i] for i, val in enumerate(row)]
            results.append(row)
        return results


__all__ = ["FastBaseEstimatorGen224", "SamplerQNN"]
