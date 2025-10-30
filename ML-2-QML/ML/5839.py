"""Hybrid estimator for classical neural networks with optional shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """Unified estimator that can evaluate a PyTorch model or a quantum circuit.

    Parameters
    ----------
    model : nn.Module | None
        Classical neural network. If ``None``, the estimator expects a quantum circuit
        passed via ``circuit`` argument.
    circuit : object | None
        Quantum circuit object. Must expose ``assign_parameters`` and ``run`` methods.
    shots : int | None
        Number of shots for quantum evaluation. If ``None``, deterministic expectation
        values are returned.
    seed : int | None
        Random seed for Gaussian noise simulation.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        circuit: Optional[object] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if model is None and circuit is None:
            raise ValueError("Either a torch model or a quantum circuit must be provided.")
        self.model = model
        self.circuit = circuit
        self.shots = shots
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _evaluate_model(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    def _evaluate_circuit(
        self,
        observables: Iterable[object],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        if self.circuit is None:
            raise RuntimeError("No quantum circuit supplied.")
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self.circuit.assign_parameters(
                dict(zip(self.circuit.parameters, params))
            )
            # Expect the circuit to expose a ``run`` method returning a state with
            # an ``expectation_value`` method.
            state = bound.run(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate(
        self,
        observables: Iterable,
        parameter_sets: Sequence[Sequence[float]],
        *,
        add_shot_noise: bool = False,
    ) -> List[List[float | complex]]:
        """Evaluate observables over parameter sets.

        When ``add_shot_noise=True`` and a quantum circuit is provided,
        shot noise is simulated by sampling from a normal distribution
        centered at the true expectation value with variance 1/shots.
        """
        if self.model is not None:
            raw = self._evaluate_model(observables, parameter_sets)
        else:
            raw = self._evaluate_circuit(observables, parameter_sets)

        if not add_shot_noise or self.shots is None:
            return raw

        noisy = []
        for row in raw:
            noisy_row = [
                float(self._rng.normal(val.real, max(1e-6, 1 / self.shots))) for val in row
            ]
            noisy.append(noisy_row)
        return noisy

    @staticmethod
    def FCL(n_features: int = 1) -> nn.Module:
        """Return a classical fully‑connected layer with a ``run`` method."""
        class FullyConnectedLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(n_features, 1)

            def run(self, thetas: Iterable[float]) -> np.ndarray:
                values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
                expectation = torch.tanh(self.linear(values)).mean(dim=0)
                return expectation.detach().numpy()

        return FullyConnectedLayer()
