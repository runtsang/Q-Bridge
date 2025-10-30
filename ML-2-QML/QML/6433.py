"""Qiskit-based hybrid estimator with advanced quantum circuit design and shot noise modeling."""
from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence, Tuple, Any

from qiskit import Aer, execute, transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator


class HybridFastEstimator:
    """
    Quantum estimator that evaluates expectation values for a parametrised circuit
    and can compute analytic gradients via the parameter‑shift rule.
    Supports configurable backend, shot noise, and a simple gradient‑descent
    training loop.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrised quantum circuit.
    backend : str, optional
        Backend name (e.g. 'qasm_simulator','statevector_simulator').
        Defaults to'statevector_simulator'.
    shots : int, optional
        Number of shots for QASM execution.  Ignored for state‑vector backends.
    seed : int | None, optional
        Random seed for shot noise simulation.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: str = "statevector_simulator",
        shots: int = 1024,
        seed: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend_name = backend
        self.shots = shots
        self.seed = seed

        if backend == "statevector_simulator":
            self.backend = AerSimulator(method="statevector")
        else:
            self.backend = Aer.get_backend(backend)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """
        Compute expectation values for each parameter set and observable.

        Returns
        -------
        numpy.ndarray
            Shape (n_sets, n_observables) with complex dtype.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            bound_circuit = self._bind(params)
            if isinstance(self.backend, AerSimulator) and self.backend_name == "statevector_simulator":
                state = Statevector.from_instruction(bound_circuit)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                qobj = transpile(bound_circuit, self.backend)
                job = execute(
                    qobj,
                    self.backend,
                    shots=self.shots,
                    seed_simulator=self.seed,
                )
                result = job.result()
                state = Statevector(result.get_statevector(bound_circuit))
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        return np.array(results)

    def _parameter_shift(
        self,
        observable: BaseOperator,
        params: Sequence[float],
        shift: float = np.pi / 2,
    ) -> np.ndarray:
        """
        Compute the analytic gradient of an observable w.r.t. the parameters
        using the parameter‑shift rule.
        """
        grads = np.zeros(len(params), dtype=np.complex128)

        for i, val in enumerate(params):
            shift_pos = list(params)
            shift_neg = list(params)
            shift_pos[i] += shift
            shift_neg[i] -= shift

            exp_pos = self._bind(shift_pos).expectation_value(observable)
            exp_neg = self._bind(shift_neg).expectation_value(observable)

            grads[i] = 0.5 * (exp_pos - exp_neg)

        return grads

    def gradient(
        self,
        observable: BaseOperator,
        param_values: Sequence[float],
    ) -> np.ndarray:
        """
        Public interface for analytic gradient computation.
        """
        return self._parameter_shift(observable, param_values)

    def train(
        self,
        parameter_sets: Sequence[Sequence[float]],
        targets: Sequence[float],
        observable: BaseOperator,
        lr: float = 0.01,
        epochs: int = 50,
        verbose: bool = False,
    ) -> None:
        """
        Simple gradient‑descent training loop for the circuit parameters.

        Parameters
        ----------
        parameter_sets : Sequence[Sequence[float]]
            Initial parameter sets to train.
        targets : Sequence[float]
            Desired expectation values for each set.
        observable : BaseOperator
            Observable to evaluate.
        lr : float, optional
            Learning rate.
        epochs : int, optional
            Number of training epochs.
        verbose : bool, optional
            If True, prints loss after each epoch.
        """
        params = np.array(parameter_sets, dtype=np.float64)

        for epoch in range(epochs):
            preds = np.array([self.evaluate([observable], [p])[0, 0] for p in params])
            loss = np.mean((preds - targets) ** 2)

            grads = np.array(
                [self.gradient(observable, p) for p in params]
            ).real  # take real part

            params -= lr * grads

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} – loss: {loss:.6f}")

        # Update the circuit parameters
        for i, param_list in enumerate(params):
            self._bind(param_list)  # bind to update internal state

__all__ = ["HybridFastEstimator"]
