"""Enhanced quantum estimator built on Qiskit.

The original ``FastBaseEstimator`` is extended by adding:

* Device selection (state‑vector, Aer simulator, and optional noise model)
* Shot‑based expectation value evaluation
* Automatic gradient computation using the parameter‑shift rule
* A clean API that mirrors the classical counterpart

The class remains fully compatible with the original ``evaluate`` method
while providing the new ``evaluate_gradients`` method.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.opflow import PauliSumOp, StateFn, CircuitStateFn, AerPauliExpectation

# Helper constants
PI2 = 2 * np.pi


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit.

    Parameters
    ----------
    circuit
        A ``QuantumCircuit`` with symbolic parameters.
    device
        Simulation backend: ``'statevector'`` (exact), ``'aer'`` (approximate),
        or ``'aer_noise'`` (Aer with a noise model).
    shots
        Number of circuit shots for the Aer backend.  Ignored for the
        state‑vector backend.
    noise_model
        Optional ``NoiseModel`` used when ``device='aer_noise'``.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        device: str = "statevector",
        shots: int | None = None,
        noise_model: NoiseModel | None = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._device = device
        self._shots = shots
        self._noise_model = noise_model

        if device not in {"statevector", "aer", "aer_noise"}:
            raise ValueError(f"Unsupported device: {device}")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
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
            bound_circ = self._bind(values)
            if self._device == "statevector":
                state = Statevector.from_instruction(bound_circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Aer backend
                backend = Aer.get_backend("aer_simulator")
                job = execute(
                    bound_circ,
                    backend=backend,
                    shots=self._shots or 1024,
                    noise_model=self._noise_model if self._device == "aer_noise" else None,
                )
                result = job.result()
                # Compute expectation via OpFlow
                row = []
                for obs in observables:
                    expectation = StateFn(obs) @ CircuitStateFn(bound_circ)
                    exp_val = AerPauliExpectation().convert(expectation).eval()
                    row.append(exp_val)
            results.append(row)
        return results

    def evaluate_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[complex]]]:
        """Return the gradient of each observable w.r.t. the circuit parameters.

        The gradient is computed via the parameter‑shift rule, which
        requires two evaluations per parameter.  The method is
        compatible with both the exact state‑vector and the Aer
        simulator backends.

        Returns
        -------
        List[List[List[complex]]]
            ``gradients[set_idx][obs_idx][param_idx]`` gives the partial
            derivative of ``observables[obs_idx]`` w.r.t. the
            ``param_idx``‑th circuit parameter for the ``set_idx``‑th
            parameter set.
        """
        observables = list(observables)
        gradients: List[List[List[complex]]] = []

        for values in parameter_sets:
            grad_set: List[List[complex]] = []
            for obs in observables:
                grad_vec: List[complex] = []
                for idx, _ in enumerate(self._parameters):
                    # Shift +π/2
                    shift_plus = list(values)
                    shift_plus[idx] += np.pi / 2
                    val_plus = self.evaluate([obs], [shift_plus])[0][0]
                    # Shift -π/2
                    shift_minus = list(values)
                    shift_minus[idx] -= np.pi / 2
                    val_minus = self.evaluate([obs], [shift_minus])[0][0]
                    # Parameter‑shift derivative
                    grad = 0.5 * (val_plus - val_minus)
                    grad_vec.append(grad)
                grad_set.append(grad_vec)
            gradients.append(grad_set)
        return gradients


__all__ = ["FastBaseEstimator"]
