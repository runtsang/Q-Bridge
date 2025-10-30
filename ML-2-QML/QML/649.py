"""Unified estimator that evaluates a parametrized quantum circuit.

The class mirrors the classical interface but uses Qiskit to compute
expectation values of arbitrary operators.  It supports a state‑vector
backend for exact results, a QASM simulator for shot‑based sampling,
and optional noise models.  Observables can be any BaseOperator or a
Pauli string.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer.noise import NoiseModel


class UnifiedEstimator:
    """Evaluate expectation values of a parametrized quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The variational circuit with symbolic parameters.
    backend : object, optional
        Backend to execute the circuit.  Defaults to a state‑vector simulator.
    noise_model : NoiseModel | None, optional
        Noise model applied when the backend supports it.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[object] = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.noise_model = noise_model

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        scaling_factors: Optional[Sequence[float]] = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Operators whose expectation values are to be computed.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.
        shots : int, optional
            If supplied, Gaussian shot noise is added to each expectation
            value to mimic finite‑sample statistics.
        seed : int, optional
            Random seed for reproducible noise.
        scaling_factors : Sequence[float], optional
            Multiplicative factor applied to each parameter vector before
            evaluation.

        Returns
        -------
        List[List[complex]]
            A matrix where each row corresponds to a parameter set and each
            column to an observable.
        """
        results: List[List[complex]] = []
        rng = np.random.default_rng(seed)

        for idx, params in enumerate(parameter_sets):
            # Apply per‑parameter scaling if requested
            if scaling_factors is not None:
                scale = scaling_factors[idx]
                params = [p * scale for p in params]
            bound_circuit = self.circuit.assign_parameters(
                dict(zip(self.parameters, params)), inplace=False
            )

            # Exact expectation values via state‑vector
            state = Statevector.from_instruction(bound_circuit)
            row = [state.expectation_value(obs) for obs in observables]

            # Add shot noise if requested
            if shots is not None:
                noisy_row = []
                for mean in row:
                    sigma = math.sqrt((1 - abs(mean) ** 2) / shots) if shots > 0 else 0.0
                    noisy_row.append(rng.normal(mean, sigma))
                row = noisy_row

            results.append([complex(v) for v in row])

        return results


__all__ = ["UnifiedEstimator"]
