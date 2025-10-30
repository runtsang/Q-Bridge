"""Hybrid estimator utilities built on PennyLane for variational circuits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import pennylane as qml
import numpy as np

ScalarObservable = Callable[[np.ndarray], np.ndarray | float]


class Gen200BaseEstimator:
    """Evaluate PennyLane QNode expectation values with flexible sampling.

    Parameters
    ----------
    circuit
        PennyLane QNode or a circuit‑generating function that accepts parameters.
    device
        PennyLane device name or ``qml.device`` instance; defaults to a default qubit device.
    shots
        Number of measurement shots for stochastic evaluation; ``None`` triggers state‑vector
        evaluation for exact expectation values.
    """

    def __init__(
        self,
        circuit: qml.QNode | Callable[..., qml.QNode],
        device: str | qml.devices.Device | None = None,
        shots: Optional[int] = None,
    ) -> None:
        if device is None:
            device = qml.device("default.qubit", wires=circuit.num_wires)
        elif isinstance(device, str):
            device = qml.device(device, wires=circuit.num_wires)
        self._device = device
        self._shots = shots

        if isinstance(circuit, qml.QNode):
            self._circuit = circuit
        else:
            self._circuit = circuit(self._device)

    def _evaluate_raw(
        self,
        observables: Iterable[qml.operation.Observable] | Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        resolved_obs = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            if self._shots is None:
                state = self._device.execute(self._circuit, [params], [qml.state])[0]
                row = [
                    obs(state) if callable(obs) else self._expect(state, obs)
                    for obs in resolved_obs
                ]
            else:
                samples = self._device.execute(self._circuit, [params], [qml.sample(self._shots)])[0]
                row = [
                    np.mean(samples) if callable(obs) else self._sample_expect(samples, obs)
                    for obs in resolved_obs
                ]
            results.append(row)
        return results

    def evaluate(
        self,
        observables: Iterable[qml.operation.Observable] | Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        return self._evaluate_raw(observables, parameter_sets)

    @staticmethod
    def _expect(state: np.ndarray, observable: qml.operation.Observable) -> complex:
        return (state.conj().T @ observable.matrix @ state).item()

    @staticmethod
    def _sample_expect(samples: np.ndarray, observable: qml.operation.Observable) -> complex:
        return float(np.mean(samples))


class Gen200Estimator(Gen200BaseEstimator):
    """Adds shot‑noise modeling by resampling outputs and computing statistics."""

    def evaluate(
        self,
        observables: Iterable[qml.operation.Observable] | Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        effective_shots = shots if shots is not None else self._shots
        if effective_shots is None:
            return super().evaluate(observables, parameter_sets)

        rng = np.random.default_rng(seed)
        noisy_results: List[List[complex]] = []

        for params in parameter_sets:
            batch_results = []
            for _ in range(effective_shots):
                single = super()._evaluate_raw(observables, [params])[0]
                batch_results.append(single)
            avg_row = [float(np.mean([r[i] for r in batch_results])) for i in range(len(batch_results[0]))]
            noisy_results.append(avg_row)

        return noisy_results


__all__ = ["Gen200BaseEstimator", "Gen200Estimator"]
