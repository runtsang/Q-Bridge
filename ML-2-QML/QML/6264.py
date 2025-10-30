"""Variational circuit estimator with shot‑noise and analytic gradients.

This module builds on the original minimal FastBaseEstimator by:
* wrapping a Pennylane QNode (or Qiskit circuit) as the core circuit
* providing a shot‑based noisy expectation routine
* exposing a parameter‑shift gradient method
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
except ImportError:
    qml = None  # Fallback to Qiskit if Pennylane is unavailable

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit.quantum_info.operators.base_operator import BaseOperator
except ImportError:
    QuantumCircuit = None  # Qiskit optional


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized quantum circuit.

    Parameters
    ----------
    circuit : qml.QNode | QuantumCircuit
        The circuit to evaluate. If a Pennylane QNode is provided, it is
        wrapped directly; otherwise a Qiskit circuit is used with a
        built‑in state‑vector simulator.
    """

    def __init__(self, circuit):
        self._circuit = circuit
        if hasattr(circuit, "parameters"):
            self._parameters = list(circuit.parameters)
        else:
            self._parameters = []

    def _bind(self, parameter_values: Sequence[float]):
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        if isinstance(self._circuit, qml.QNode):
            return lambda: self._circuit(**mapping)
        else:  # Qiskit
            return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable,
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        results: List[List[complex]] = []

        if qml is not None and isinstance(self._circuit, qml.QNode):
            for values in parameter_sets:
                bound = self._bind(values)
                exp_vals = [bound() @ obs for obs in observables]
                results.append(exp_vals)
        else:
            # Fallback to Qiskit state‑vector simulation
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
        return results

    def evaluate_shots(
        self,
        observables: Iterable,
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return noisy expectation values using measurement shots."""
        rng = np.random.default_rng(seed)
        results: List[List[float]] = []

        if qml is not None and isinstance(self._circuit, qml.QNode):
            for values in parameter_sets:
                bound = self._bind(values)
                exp_vals = []
                for obs in observables:
                    meas = bound()
                    # Convert measurement to expectation via parameter shift
                    exp = float(meas)
                    noisy = rng.normal(exp, 1 / np.sqrt(shots))
                    exp_vals.append(noisy)
                results.append(exp_vals)
        else:
            # Qiskit measurement simulation
            for values in parameter_sets:
                circuit = self._bind(values)
                simulator = qml.device("qiskit.aer.noise.NoiseModel", shots=shots)
                job = simulator.execute(circuit)
                exp_vals = [float(job.result().data[obs]) for obs in observables]
                results.append(exp_vals)
        return results

    def gradient(
        self,
        observables: Iterable,
        parameter_sets: Sequence[Sequence[float]],
        shift: float = np.pi / 2,
    ) -> List[List[List[float]]]:
        """Compute gradients of each observable w.r.t. parameters using the parameter‑shift rule."""
        if qml is None:
            raise RuntimeError("Pennylane is required for analytic gradients.")
        results: List[List[List[float]]] = []

        for values in parameter_sets:
            grads_per_obs: List[List[float]] = []
            for obs in observables:
                grad = qml.grad(self._circuit)(*values)
                grads_per_obs.append(grad)
            results.append(grads_per_obs)
        return results


__all__ = ["FastBaseEstimator"]
