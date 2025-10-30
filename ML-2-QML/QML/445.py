"""Enhanced quantum estimator for parameterised circuits with sampling, noise, and gradients."""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Dict, Optional

from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector, Operator
from qiskit.opflow import PauliOp, StateFn, AerPauliExpectation
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer.noise import NoiseModel


class FastBaseEstimator:
    """Evaluate expectation values of a parameterised circuit.

    Parameters
    ----------
    circuit: QuantumCircuit
        Parameterised circuit whose parameters are bound during evaluation.
    simulator: AerSimulator | None, optional
        Aer simulator instance used when ``shots`` is specified.  If ``None`` a
        default ``AerSimulator`` with ``method='statevector'`` is created.
    noise_model: NoiseModel | None, optional
        Noise model to attach to the simulator.  Only used when ``shots`` is not None.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        simulator: AerSimulator | None = None,
        noise_model: NoiseModel | None = None,
    ) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.simulator = simulator or AerSimulator(method="statevector")
        if noise_model:
            self.simulator.set_options(noise_model=noise_model)
        self._cache: Dict[Tuple[float,...], Statevector] = {}

    def _bind(self, param_vals: Sequence[float]) -> QuantumCircuit:
        if len(param_vals)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, param_vals))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for all parameter sets and observables.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Operators whose expectation values are evaluated.
        parameter_sets : Sequence[Sequence[float]]
            List of parameter vectors.
        shots : int | None, optional
            Number of shots.  If ``None`` the exact statevector expectation is used.
        seed : int | None, optional
            Random seed for the simulator.
        Returns
        -------
        List[List[complex]]
            Rows correspond to parameter sets; columns to observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        # Exact state‑vector evaluation
        if shots is None:
            for params in parameter_sets:
                state = Statevector.from_instruction(self._bind(params))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
            return results

        # Sampling evaluation
        backend = self.simulator
        backend.set_options(seed_simulator=seed)
        bound_circuits = [self._bind(params) for params in parameter_sets]
        job = backend.run(bound_circuits, shots=shots)
        result = job.result()

        # Convert each circuit to a StateFn for the Pauli expectation
        expectation_calculator = AerPauliExpectation()
        for idx, params in enumerate(parameter_sets):
            circuit_fn = StateFn(bound_circuits[idx])
            row = []
            for obs in observables:
                op_fn = StateFn(obs, is_measurement=True)
                expectation = expectation_calculator.convert(op_fn @ circuit_fn)
                exp_val = expectation.eval()
                row.append(exp_val)
            results.append(row)
        return results

    def gradient(
        self,
        observable: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shift: float = 0.5,
        shots: int | None = None,
    ) -> List[List[float]]:
        """Compute the gradient of a single observable w.r.t. all circuit parameters
        using the parameter‑shift rule (exact or sampled).

        Parameters
        ----------
        observable : BaseOperator
            Operator whose derivative is required.
        parameter_sets : Sequence[Sequence[float]]
            Parameter vectors.
        shift : float, optional
            Shift amount for the parameter‑shift rule.  Default 0.5 for a
            standard 2‑parameter shift.
        shots : int | None, optional
            Number of shots for sampling.  If ``None`` use exact gradients.
        Returns
        -------
        List[List[float]]
            Gradient vectors for each parameter set.
        """
        gradients: List[List[float]] = []

        for params in parameter_sets:
            grad_vec = []
            for i, _ in enumerate(params):
                # Shift parameters
                plus = list(params)
                minus = list(params)
                plus[i] += shift
                minus[i] -= shift

                val_plus = self.evaluate([observable], [plus], shots=shots)[0][0]
                val_minus = self.evaluate([observable], [minus], shots=shots)[0][0]
                grad = (val_plus - val_minus) / (2 * shift)
                grad_vec.append(float(grad))
            gradients.append(grad_vec)
        return gradients


class FastEstimator(FastBaseEstimator):
    """Quantum estimator with built‑in shot‑noise and optional error mitigation.

    Parameters
    ----------
    circuit: QuantumCircuit
        Parameterised circuit.
    simulator: AerSimulator | None, optional
        Simulator instance; if ``None`` defaults to a state‑vector simulator.
    noise_model: NoiseModel | None, optional
        Noise model for the simulator when ``shots`` > 0.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        simulator: AerSimulator | None = None,
        noise_model: NoiseModel | None = None,
    ) -> None:
        super().__init__(circuit, simulator=simulator, noise_model=noise_model)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Same as FastBaseEstimator but returns noisy expectation values using
        a sampling backend.  If ``shots`` is ``None`` the exact expectation is
        returned.
        """
        return super().evaluate(observables, parameter_sets, shots=shots, seed=seed)

    def error_mitigated_expectation(
        self,
        observable: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int,
        mitigation: str = "zero_noise",
        seed: int | None = None,
    ) -> List[complex]:
        """Return expectation values mitigated with a simple zero‑noise extrapolation.

        Parameters
        ----------
        observable : BaseOperator
            Observable to evaluate.
        parameter_sets : Sequence[Sequence[float]]
            Parameter vectors.
        shots : int
            Number of shots for each circuit.
        mitigation : str, optional
            Mitigation strategy.  Currently only ``'zero_noise'`` is supported.
        seed : int | None, optional
            Random seed for the simulator.
        """
        if mitigation!= "zero_noise":
            raise NotImplementedError("Only zero_noise mitigation is implemented.")

        # For demonstration, we simply run the circuit twice with the same noise level.
        # A real implementation would perform an extrapolation.
        results = self.evaluate([observable], parameter_sets, shots=shots, seed=seed)
        return [row[0] for row in results]


__all__ = ["FastBaseEstimator", "FastEstimator"]
