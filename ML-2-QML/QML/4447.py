"""Quantum‑centric hybrid estimator mirroring the classical implementation.

The QML version keeps the same public API but prioritises quantum back‑ends.
It can evaluate Qiskit circuits, TorchQuantum modules, quantum kernels,
quantum LSTMs, and, if desired, classical neural nets.  The shot‑noise
simulation remains identical to the classical counterpart.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable, Union, Optional

# Optional quantum back‑ends --------------------------------------------------
try:
    from qiskit.circuit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit.quantum_info.operators.base_operator import BaseOperator
except Exception:  # pragma: no cover
    QuantumCircuit = None  # type: ignore
    Statevector = None  # type: ignore
    BaseOperator = None  # type: ignore

try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except Exception:  # pragma: no cover
    tq = None  # type: ignore
    tqf = None  # type: ignore

# Optional higher‑level modules ------------------------------------------------
try:
    from QTransformerTorch import TextClassifier
except Exception:  # pragma: no cover
    TextClassifier = None  # type: ignore

try:
    from QuantumKernelMethod import Kernel
except Exception:  # pragma: no cover
    Kernel = None  # type: ignore

try:
    from QLSTM import QLSTM
except Exception:  # pragma: no cover
    QLSTM = None  # type: ignore

# ---------------------------------------------------------------------------

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """Quantum‑centric estimator with the same API as the classical version."""

    def __init__(
        self,
        model: Union[
            nn.Module,
            "QuantumCircuit",
            "tq.QuantumModule",
            "Kernel",
            "QLSTM",
            "TextClassifier",
        ],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.shots = shots
        self.seed = seed
        self.rng = np.random.default_rng(seed) if shots is not None else None

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | Iterable[BaseOperator] | None = None,
        parameter_sets: Sequence[Sequence[float]] | Sequence[Sequence[torch.Tensor]] | None = None,
    ) -> List[List[float]]:
        """Dispatch evaluation to the appropriate backend based on model type."""
        if isinstance(self.model, QuantumCircuit):
            return self._evaluate_qiskit(observables, parameter_sets)
        if tq is not None and isinstance(self.model, tq.QuantumModule):
            return self._evaluate_tq(observables, parameter_sets)
        if isinstance(self.model, Kernel):
            return self._evaluate_kernel(observables, parameter_sets)
        if isinstance(self.model, QLSTM):
            return self._evaluate_lstm(observables, parameter_sets)
        if isinstance(self.model, TextClassifier):
            return self._evaluate_classifier(observables, parameter_sets)
        if isinstance(self.model, nn.Module):
            return self._evaluate_nn(observables, parameter_sets)
        raise TypeError(f"Unsupported model type: {type(self.model)}")

    # ----------------------------------------------------------------------- #
    # Backend specific evaluators
    # ----------------------------------------------------------------------- #
    def _evaluate_qiskit(
        self,
        observables: Iterable[BaseOperator] | None,
        parameter_sets: Sequence[Sequence[float]] | None,
    ) -> List[List[float]]:
        if QuantumCircuit is None:  # pragma: no cover
            raise RuntimeError("Qiskit is not available.")
        observables = list(observables or [])
        results: List[List[float]] = []
        for values in parameter_sets or []:
            bound = self.model.assign_parameters(dict(zip(self.model.parameters, values)), inplace=False)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(o) for o in observables]
            results.append([float(v.real) for v in row])
        return self._add_noise(results)

    def _evaluate_tq(
        self,
        observables: Iterable[ScalarObservable] | None,
        parameter_sets: Sequence[Sequence[torch.Tensor]] | None,
    ) -> List[List[float]]:
        if tq is None:  # pragma: no cover
            raise RuntimeError("TorchQuantum is not available.")
        observables = list(observables or [])
        results: List[List[float]] = []
        for params in parameter_sets or []:
            if isinstance(params, Sequence) and all(isinstance(p, torch.Tensor) for p in params):
                outputs = self.model(*params)
            else:
                outputs = self.model(params[0])
            row = [float(o) for o in outputs]
            results.append(row)
        return self._add_noise(results)

    def _evaluate_kernel(
        self,
        observables: Iterable[ScalarObservable] | None,
        parameter_sets: Sequence[Sequence[torch.Tensor]] | None,
    ) -> List[List[float]]:
        if Kernel is None:  # pragma: no cover
            raise RuntimeError("QuantumKernelMethod is not available.")
        kernel = self.model
        results: List[List[float]] = []
        for x in parameter_sets or []:
            row = [float(kernel(x, y)) for y in kernel.train_data]
            results.append(row)
        return self._add_noise(results)

    def _evaluate_lstm(
        self,
        observables: Iterable[ScalarObservable] | None,
        parameter_sets: Sequence[Sequence[torch.Tensor]] | None,
    ) -> List[List[float]]:
        if QLSTM is None:  # pragma: no cover
            raise RuntimeError("QLSTM is not available.")
        results: List[List[float]] = []
        for seq in parameter_sets or []:
            seq_tensor = torch.stack(seq)
            seq_tensor = seq_tensor.unsqueeze(1)
            outputs, _ = self.model(seq_tensor)
            last = outputs[-1].squeeze(0)
            row = [float(last[i].item()) for i in range(last.shape[0])]
            results.append(row)
        return self._add_noise(results)

    def _evaluate_classifier(
        self,
        observables: Iterable[ScalarObservable] | None,
        parameter_sets: Sequence[Sequence[int]] | None,
    ) -> List[List[float]]:
        if TextClassifier is None:  # pragma: no cover
            raise RuntimeError("TextClassifier is not available.")
        results: List[List[float]] = []
        for seq in parameter_sets or []:
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            logits = self.model(seq_tensor)
            probs = logits.softmax(dim=-1)
            row = probs.tolist()
            results.append(row)
        return self._add_noise(results)

    def _evaluate_nn(
        self,
        observables: Iterable[ScalarObservable] | None,
        parameter_sets: Sequence[Sequence[float]] | None,
    ) -> List[List[float]]:
        if nn is None:  # pragma: no cover
            raise RuntimeError("PyTorch is not available.")
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets or []:
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
        return self._add_noise(results)

    # ----------------------------------------------------------------------- #
    # Utility
    # ----------------------------------------------------------------------- #
    def _add_noise(self, results: List[List[float]]) -> List[List[float]]:
        if self.shots is None:
            return results
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridEstimator"]
