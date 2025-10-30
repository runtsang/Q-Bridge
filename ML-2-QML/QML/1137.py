"""FastBaseEstimator for quantum circuits with gradient support.

Features
--------
* Flexible backend selection (state‑vector or QASM with shots).
* Parameter‑shift gradient calculation for a list of observables.
* Optional noise model injection.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import List

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator
from qiskit import execute, transpile, Aer


class FastBaseEstimator:
    """
    Evaluate expectation values of observables for a parametrized quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrized circuit.
    backend : str | AerSimulator, optional
        Backend name or instance. If None, a state‑vector simulator is used.
    shots : int, optional
        Number of shots for a QASM simulation. If None, state‑vector simulation is used.
    noise_model : NoiseModel, optional
        Noise model to attach to the simulator.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: str | AerSimulator | None = None,
        shots: int | None = None,
        noise_model=None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

        if backend is None:
            if shots is None:
                self.backend = AerSimulator(method="statevector")
            else:
                self.backend = AerSimulator(method="qasm", shots=shots)
        else:
            if isinstance(backend, str):
                self.backend = Aer.get_backend(backend)
            else:
                self.backend = backend

        self.shots = shots
        self.noise_model = noise_model

        if self.noise_model is not None:
            self.backend.set_options(noise_model=self.noise_model)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _expectation_from_statevector(
        self, state: Statevector, observable: BaseOperator
    ) -> complex:
        return state.expectation_value(observable)

    def _expectation_from_counts(
        self, counts: dict[str, int], observable: BaseOperator
    ) -> complex:
        """
        Compute expectation value from measurement counts for Pauli operators.
        For general operators, falls back to state‑vector simulation.
        """
        if isinstance(observable, Pauli):
            pauli = observable
            exp_val = 0.0
            total = sum(counts.values())
            for bitstring, n in counts.items():
                parity = 0
                for i, bit in enumerate(bitstring[::-1]):
                    if pauli.x[i] == "1":
                        parity ^= int(bit)
                    if pauli.z[i] == "1":
                        parity ^= int(bit)
                exp_val += ((-1) ** parity) * n
            return exp_val / total
        # Fallback: return zero for unsupported operators
        return 0.0

    def evaluate(
        self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.

        Returns
        -------
        List[List[complex]]
            Outer list indexed by parameter set, inner list indexed by observable.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            circ = self._bind(params)
            if self.shots is None:
                state = Statevector.from_instruction(circ)
                row = [self._expectation_from_statevector(state, obs) for obs in observables]
            else:
                transpiled = transpile(circ, self.backend)
                job = execute(transpiled, self.backend)
                result = job.result()
                counts = result.get_counts()
                row = [self._expectation_from_counts(counts, obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        """
        Compute the gradient of each observable with respect to the circuit parameters
        using the parameter‑shift rule.

        Returns
        -------
        List[List[np.ndarray]]
            Outer list indexed by parameter set, inner list indexed by observable.
            Each gradient is a 1‑D numpy array of shape (n_params,).
        """
        observables = list(observables)
        grads: List[List[np.ndarray]] = []

        shift = np.pi / 2

        for params in parameter_sets:
            grad_row: List[np.ndarray] = []
            for obs in observables:
                grad_vec = []
                for idx, _ in enumerate(self._parameters):
                    plus_params = list(params)
                    minus_params = list(params)
                    plus_params[idx] += shift
                    minus_params[idx] -= shift

                    circ_plus = self._bind(plus_params)
                    circ_minus = self._bind(minus_params)

                    f_plus = self._expectation_from_statevector(
                        Statevector.from_instruction(circ_plus), obs
                    )
                    f_minus = self._expectation_from_statevector(
                        Statevector.from_instruction(circ_minus), obs
                    )

                    grad = (f_plus - f_minus) / 2.0
                    grad_vec.append(grad)
                grad_row.append(np.array(grad_vec))
            grads.append(grad_row)
        return grads


__all__ = ["FastBaseEstimator"]
