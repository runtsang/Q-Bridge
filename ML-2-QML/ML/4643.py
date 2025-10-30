"""Hybrid estimator that can wrap a PyTorch nn.Module or a Qiskit QuantumCircuit."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union, Tuple, Any

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFastEstimator:
    """Unified estimator for classical and quantum models.

    Parameters
    ----------
    model
        Either a :class:`torch.nn.Module` (classical) or a
        :class:`qiskit.circuit.QuantumCircuit` (quantum).
    shots
        Number of shots for quantum simulation or Gaussian noise variance
        scaling for classical model.  If ``None`` no noise is added.
    seed
        Random seed for reproducibility.  Only used when ``shots`` is
        supplied.
    """

    def __init__(
        self,
        model: Union[nn.Module, "QuantumCircuit"],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.shots = shots
        self.seed = seed
        if isinstance(model, nn.Module):
            self._is_quantum = False
        else:
            self._is_quantum = True
            from qiskit import Aer, execute
            self.backend = Aer.get_backend("qasm_simulator")

    def evaluate(
        self,
        observables: Iterable[Any],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute values for each observable and parameter set.

        For a classical model the observables are callables on the
        network output.  For a quantum model they are
        :class:`qiskit.quantum_info.operators.base_operator.BaseOperator`
        objects.  The ``shots`` and ``seed`` arguments override the
        instance defaults only for the current call.
        """
        if shots is None:
            shots = self.shots
        if seed is None:
            seed = self.seed

        if self._is_quantum:
            return self._eval_quantum(observables, parameter_sets, shots)
        return self._eval_classical(observables, parameter_sets, shots, seed)

    def _eval_classical(
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
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def _eval_quantum(
        self,
        observables: Iterable["BaseOperator"],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None,
    ) -> List[List[float]]:
        from qiskit.quantum_info import Statevector
        observables = list(observables)
        results: List[List[float]] = []

        for values in parameter_sets:
            bound = self.model.assign_parameters(dict(zip(self.model.parameters, values)), inplace=False)
            state = Statevector.from_instruction(bound)
            row = [float(state.expectation_value(obs)) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        # Simple shot-noise simulation: re-evaluate each expectation with shot sampling
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(np.random.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridFastEstimator"]
