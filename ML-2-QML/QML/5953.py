"""Enhanced estimator for Qiskit circuits with shots, noise, and gradient support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.quantum_info import Statevector, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer.noise import NoiseModel


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The parametrized circuit to evaluate.
    backend : Backend, optional
        Aer simulator backend.  Defaults to ``Aer.get_backend('statevector_simulator')``.
    noise_model : NoiseModel | None, optional
        Noise model to apply when using a noisy simulator.  If ``None`` the
        simulator is noise‑free.
    shots : int | None, optional
        Number of shots for the measurement circuit.  If ``None`` the statevector
        simulator is used.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[Backend] = None,
        noise_model: Optional[NoiseModel] = None,
        shots: Optional[int] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.noise_model = noise_model
        self.shots = shots
        if shots is None:
            self.backend = Aer.get_backend("statevector_simulator")
        else:
            self.backend = Aer.get_backend("qasm_simulator")
            if noise_model:
                self.backend.set_options(noise_model=noise_model)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with parameters bound to the supplied values."""
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            if self.shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Measure all qubits to compute expectation values
                meas_circ = bound.copy()
                meas_circ.measure_all()
                job = self.backend.run(meas_circ, shots=self.shots, noise_model=self.noise_model)
                counts = job.result().get_counts()
                probs = {tuple(map(int, k[::-1])): v / self.shots for k, v in counts.items()}
                row = [self._expectation_from_counts(obs, probs) for obs in observables]
            results.append(row)
        return results

    def _expectation_from_counts(self, obs: BaseOperator, probs: dict) -> complex:
        """Compute expectation value from measurement probabilities."""
        # Convert observable to a matrix
        mat = Operator(obs).data
        # Build the expectation as a weighted sum over basis states
        exp = 0j
        for state, p in probs.items():
            vec = np.array([1 if bit else 0 for bit in state], dtype=complex)
            exp += p * vec.conj().T @ mat @ vec
        return exp

    def gradient(
        self,
        observable: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute the gradient of a single observable using the parameter‑shift rule.

        Returns a list of gradients for each parameter in each parameter set.
        """
        shift = np.pi / 2
        gradients: List[List[float]] = []

        for params in parameter_sets:
            grad_row: List[float] = []
            for i in range(len(params)):
                shifted_plus = list(params)
                shifted_minus = list(params)
                shifted_plus[i] += shift
                shifted_minus[i] -= shift
                exp_plus = self.evaluate([observable], [shifted_plus])[0][0]
                exp_minus = self.evaluate([observable], [shifted_minus])[0][0]
                grad = (exp_plus - exp_minus).real / 2
                grad_row.append(grad)
            gradients.append(grad_row)
        return gradients


__all__ = ["FastBaseEstimator"]
