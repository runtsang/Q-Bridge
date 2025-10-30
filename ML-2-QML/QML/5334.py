"""Hybrid quantum estimator that mimics the classical EstimatorQNN__gen223 interface.

The implementation combines:
- A square input filter encoded into a random quantum circuit (Conv.py).
- Weight parameters applied after the random layer (QuantumNAT).
- Pauli‑Z measurement on all qubits.
- Optional Gaussian shot noise to emulate finite‑shot sampling (FastEstimator).

The class can be used as a drop‑in replacement for the classical model in
experiments that require a quantum backend.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.providers.aer import AerSimulator
from qiskit.primitives import StatevectorEstimator
from collections.abc import Iterable, Sequence
from typing import List

# ---------------------------------------------------------------------------

class EstimatorQNN__gen223:
    """
    Quantum estimator that accepts the same interface as the classical
    EstimatorQNN__gen223.

    Parameters
    ----------
    kernel_size : int, default 2
        Determines the number of qubits (kernel_size²) and the shape of
        the input data.
    backend : Backend | None, default None
        Quantum backend used by the StatevectorEstimator.  If None a
        state‑vector simulator is used.
    shots : int | None, default 100
        Number of shots for the Gaussian noise model.
    threshold : float, default 0.5
        Threshold used when binding classical data to the input parameters.
    seed : int | None, default None
        Random seed for the Gaussian noise.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int | None = 100,
        threshold: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.threshold = threshold
        self.seed = seed

        # Build parameterised circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        self.input_params = [Parameter(f"input{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.input_params[i], i)
        self.circuit.barrier()

        # Random layer (QuantumNAT style)
        self.circuit += random_circuit(self.n_qubits, 2)

        # Weight parameters
        self.weight_params = [Parameter(f"weight{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.weight_params[i], i)

        self.circuit.measure_all()

        # Observable: Pauli‑Z on every qubit (averaged)
        self._observable = SparsePauliOp.from_list([("Z" * self.n_qubits, 1)])

        # Estimator primitive
        self.estimator = StatevectorEstimator(backend=self.backend)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------------

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        """Return a circuit with the given parameter values bound."""
        if len(values)!= len(self.input_params) + len(self.weight_params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.input_params + self.weight_params, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    # -----------------------------------------------------------------------

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            If empty the default Pauli‑Z observable is used.
        parameter_sets : Sequence[Sequence[float]]
            Each inner sequence contains values for all input and weight
            parameters in the order ``input_params + weight_params``.
        shots : int | None, optional
            Override the default shot count for the noise model.
        seed : int | None, optional
            Override the random seed for the noise model.

        Returns
        -------
        List[List[complex]]
            Matrix of expectation values.
        """
        observables = list(observables) or [self._observable]
        shots = shots if shots is not None else self.shots
        results: List[List[complex]] = []

        for params in parameter_sets:
            bound = self._bind(params)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        # Add Gaussian shot noise if requested
        if shots is not None:
            rng = np.random.default_rng(seed if seed is not None else self.seed)
            noisy: List[List[complex]] = []
            for row in results:
                noisy_row = [
                    complex(
                        rng.normal(mean.real, max(1e-6, 1 / shots)),
                        rng.normal(mean.imag, max(1e-6, 1 / shots)),
                    )
                    for mean in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = ["EstimatorQNN__gen223"]
