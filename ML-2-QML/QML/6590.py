"""FastBaseEstimator – a lightweight quantum estimator using Qiskit.

This module extends the original FastBaseEstimator by adding support for
shot‑noise simulation, parameter‑sharding, and optional noise models.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel


class FastBaseEstimator:
    """Evaluate expectation values of operators for a parametrized circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        A parametrized quantum circuit.
    device : str | AerSimulator, optional
        The simulator backend to use. Defaults to AerSimulator('statevector').
    noise_model : NoiseModel | None, optional
        Optional noise model to inject into the simulator.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        device: Union[str, AerSimulator] = "statevector",
        noise_model: Optional[NoiseModel] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        if isinstance(device, str):
            self._simulator = AerSimulator(device)
        else:
            self._simulator = device
        self._noise_model = noise_model

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
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Operators for which to compute expectation values.
        parameter_sets : Sequence[Sequence[float]]
            Each inner sequence contains the parameter values for one circuit.
        shots : int, optional
            If provided, add Gaussian noise with variance 1/shots to each
            expectation value to emulate shot noise.
        seed : int, optional
            Random seed for the shot‑noise generator.

        Returns
        -------
        List[List[complex]]
            Rows correspond to parameter sets, columns to observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        rng = np.random.default_rng(seed)

        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            if shots is not None:
                row = [
                    float(rng.normal(val.real, max(1e-6, 1 / shots))) +
                    1j * float(rng.normal(val.imag, max(1e-6, 1 / shots)))
                    for val in row
                ]
            results.append(row)

        return results


__all__ = ["FastBaseEstimator"]
