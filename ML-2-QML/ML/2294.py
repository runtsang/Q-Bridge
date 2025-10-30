from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable, Union, Any

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

class HybridBaseEstimator:
    """Unified estimator supporting PyTorch models and Qiskit quantum circuits.

    The class accepts either a torch.nn.Module or a qiskit.circuit.QuantumCircuit
    during initialization.  Observables are interpreted accordingly:
      * For torch models: callables that return a Tensor or scalar.
      * For quantum circuits: qiskit.quantum_info.operators.BaseOperator instances.

    The evaluate method returns a list of lists of results for each parameter set.
    Optional shot noise can be added for deterministic models.
    """

    def __init__(self, model: Union[nn.Module, Any]) -> None:
        self.model = model
        # Determine if the model is a quantum circuit by checking for Qiskit-specific
        # attributes.  The check is deliberately lightweight to avoid importing Qiskit
        # unless necessary.
        self.is_quantum = hasattr(model, "assign_parameters") and hasattr(model, "parameters")

    @staticmethod
    def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        observables: Iterable[Union[ScalarObservable, Any]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float | complex]]:
        if self.is_quantum:
            return self._evaluate_quantum(observables, parameter_sets, shots, seed)
        return self._evaluate_classical(observables, parameter_sets, shots, seed)

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
                inputs = self._ensure_batch(params)
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
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def _evaluate_quantum(
        self,
        observables: Iterable[Any],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None,
        seed: int | None,
    ) -> List[List[complex]]:
        try:
            from qiskit.quantum_info import Statevector
        except ImportError as exc:
            raise RuntimeError("Qiskit is required for quantum evaluation.") from exc

        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_circ = self.model.assign_parameters(dict(zip(self.model.parameters, values)), inplace=False)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                rng.normal(row[i].real, 1 / np.sqrt(shots)) + 1j * rng.normal(row[i].imag, 1 / np.sqrt(shots))
                for i in range(len(row))
            ]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridBaseEstimator"]
