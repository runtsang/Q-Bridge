"""Fast quantum estimator with shot noise and gradient support using Qiskit Aer."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Dict, Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Operator, Pauli
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit.

    Features
    --------
    * Exact statevector evaluation for noiseless circuits.
    * Shot-noise simulation via AerSimulator with configurable shots.
    * Gradient estimation using the parameter-shift rule.
    * Optional noise models for realistic device simulation.
    """

    def __init__(self, circuit: QuantumCircuit, noise_model: NoiseModel | None = None):
        self._base_circuit = circuit
        self._parameters = list(circuit.parameters)
        self._noise_model = noise_model
        self._simulator = AerSimulator(noise_model=noise_model, seed_simulator=None)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._base_circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Operator] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
    ) -> List[List[complex]]:
        """Exact expectation values via statevector simulation."""
        if observables is None:
            observables = [Operator(np.eye(2 ** self._base_circuit.num_qubits))]
        if parameter_sets is None:
            parameter_sets = []

        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_circ = self._bind(values)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[Pauli] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int = 1024,
    ) -> List[List[float]]:
        """Sample measurement outcomes to estimate expectation values."""
        if observables is None:
            observables = [Pauli("I" * self._base_circuit.num_qubits)]
        if parameter_sets is None:
            parameter_sets = []

        results: List[List[float]] = []

        for values in parameter_sets:
            bound_circ = self._bind(values)
            bound_circ = transpile(bound_circ, self._simulator)
            job = self._simulator.run(bound_circ, shots=shots)
            result = job.result()
            counts = result.get_counts(bound_circ)
            # convert counts to expectation value for each observable
            row = []
            for obs in observables:
                exp_val = 0.0
                for bitstring, freq in counts.items():
                    parity = 1
                    for qubit, pauli in enumerate(reversed(obs.to_label())):
                        if pauli == "Z":
                            parity *= 1 if bitstring[qubit] == "0" else -1
                    exp_val += parity * freq / shots
                row.append(exp_val)
            results.append(row)
        return results

    def evaluate_with_grad(
        self,
        observables: Iterable[Pauli] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shift: float = 2.0,
        shots: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Compute gradients of expectation values via the parameter-shift rule.

        Parameters
        ----------
        shift : float
            Shift angle used in the parameter-shift rule.
        shots : int, optional
            If provided, gradients are estimated using the same shot-based sampler.
        """
        if observables is None:
            observables = [Pauli("I" * self._base_circuit.num_qubits)]
        if parameter_sets is None:
            parameter_sets = []

        results: List[Dict[str, Any]] = []

        for values in parameter_sets:
            grads: List[List[float]] = []
            for idx, param in enumerate(self._parameters):
                # shift forward
                forward_vals = list(values)
                forward_vals[idx] += shift / 2
                forward = self._bind(forward_vals)
                # shift backward
                backward_vals = list(values)
                backward_vals[idx] -= shift / 2
                backward = self._bind(backward_vals)

                if shots is None:
                    # exact expectation
                    f_state = Statevector.from_instruction(forward)
                    b_state = Statevector.from_instruction(backward)
                    f_vals = [f_state.expectation_value(obs) for obs in observables]
                    b_vals = [b_state.expectation_value(obs) for obs in observables]
                else:
                    # shot-based estimation
                    f_job = self._simulator.run(forward, shots=shots)
                    b_job = self._simulator.run(backward, shots=shots)
                    f_counts = f_job.result().get_counts(forward)
                    b_counts = b_job.result().get_counts(backward)

                    def exp_from_counts(counts):
                        exp_val = 0.0
                        for bitstring, freq in counts.items():
                            parity = 1
                            for qubit, pauli in enumerate(reversed(obs.to_label())):
                                if pauli == "Z":
                                    parity *= 1 if bitstring[qubit] == "0" else -1
                            exp_val += parity * freq / shots
                        return exp_val

                    f_vals = [exp_from_counts(f_counts) for obs in observables]
                    b_vals = [exp_from_counts(b_counts) for obs in observables]

                grad = [(f - b) / (2 * np.sin(shift / 2)) for f, b in zip(f_vals, b_vals)]
                grads.append(grad)
            results.append({"gradients": grads})
        return results


__all__ = ["FastBaseEstimator"]
