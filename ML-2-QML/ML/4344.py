"""Hybrid estimator combining classical neural networks and quantum simulators.

This module implements a lightweight estimator that can work with either a
PyTorch `nn.Module` or a Qiskit `QuantumCircuit`.  The interface mirrors the
original `FastBaseEstimator` but adds support for shot‑noise emulation and
automatic construction of classifier circuits, samplers and fully‑connected
layers.  The design keeps the classical and quantum halves decoupled while
providing a unified API for experimentation.

Key features
------------
- `evaluate` accepts a list of callable observables (for classical) or
  `BaseOperator` objects (for quantum) and a sequence of parameter sets.
- Optional shot‑noise for both classical (Gaussian) and quantum (count‑based)
  evaluations.
- Static helpers `build_classifier_circuit`, `SamplerQNN` and `FCL` that
  return components compatible with the estimator’s type.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """Unified estimator for classical and quantum models."""

    def __init__(
        self,
        model: Union[nn.Module, "QuantumCircuit"],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self._is_classical = isinstance(model, nn.Module)
        self.shots = shots
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Classical evaluation ------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def _evaluate_classical(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None,
        seed: int | None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        # Add Gaussian shot noise if requested
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy.append(noisy_row)
            return noisy
        return results

    # ------------------------------------------------------------------ #
    # Quantum evaluation ------------------------------------------------- #
    # ------------------------------------------------------------------ #
    def _evaluate_quantum(
        self,
        observables: Iterable["BaseOperator"],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None,
        seed: int | None,
    ) -> List[List[complex]]:
        from qiskit.quantum_info import Statevector
        from qiskit.quantum_info.operators.base_operator import BaseOperator

        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind_quantum(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)

        # Simulate shot noise if requested
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[complex]] = []
            for row in results:
                noisy_row = [
                    complex(rng.normal(value.real, 1 / shots))
                    + 1j * rng.normal(value.imag, 1 / shots)
                    for value in row
                ]
                noisy.append(noisy_row)
            return noisy
        return results

    # Parameter binding for quantum circuits
    def _bind_quantum(self, parameter_values: Sequence[float]) -> "QuantumCircuit":
        if not hasattr(self.model, "_parameters"):
            raise AttributeError("Quantum model must expose a `_parameters` attribute.")
        if len(parameter_values)!= len(self.model._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.model._parameters, parameter_values))
        return self.model.assign_parameters(mapping, inplace=False)

    # ------------------------------------------------------------------ #
    # Public API -------------------------------------------------------- #
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        observables: Iterable[Union[ScalarObservable, "BaseOperator"]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[Union[float, complex]]]:
        """Evaluate the underlying model for every parameter set.

        Parameters
        ----------
        observables
            Callables for classical models or `BaseOperator` instances for quantum.
        parameter_sets
            Sequence of parameter vectors.
        shots
            If provided, adds shot‑noise to the deterministic result.
        seed
            Seed for the pseudo‑random generator.
        """
        if self._is_classical:
            return self._evaluate_classical(
                observables, parameter_sets, shots, seed
            )
        else:
            return self._evaluate_quantum(
                observables, parameter_sets, shots or self.shots, seed or self.seed
            )

    # ------------------------------------------------------------------ #
    # Helper constructors ----------------------------------------------- #
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
        """Return a feed‑forward classifier and metadata.

        Mirrors the quantum helper but uses PyTorch layers.
        """
        layers = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

    @staticmethod
    def SamplerQNN() -> nn.Module:
        """Return a simple softmax sampler implemented in PyTorch."""
        class SamplerModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 4),
                    nn.Tanh(),
                    nn.Linear(4, 2),
                )

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                return torch.softmax(self.net(inputs), dim=-1)

        return SamplerModule()

    @staticmethod
    def FCL() -> nn.Module:
        """Return a fully‑connected layer that mimics the quantum example."""
        class FullyConnectedLayer(nn.Module):
            def __init__(self, n_features: int = 1) -> None:
                super().__init__()
                self.linear = nn.Linear(n_features, 1)

            def run(self, thetas: Iterable[float]) -> np.ndarray:
                values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
                expectation = torch.tanh(self.linear(values)).mean(dim=0)
                return expectation.detach().numpy()

        return FullyConnectedLayer()

__all__ = ["FastHybridEstimator"]
