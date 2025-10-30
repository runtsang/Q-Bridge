"""Unified estimator and LSTM that can operate in classical or quantum mode.

The module contains two main components:
* :class:`UnifiedBaseEstimator` – a hybrid estimator that can evaluate a PyTorch model or a Qiskit circuit.
  It supports batched parameter sets, multiple observables and optional Gaussian shot noise.
* :class:`UnifiedLSTMTagger` – a single LSTMTagger that accepts a ``n_qubits`` argument.
  When ``n_qubits>0`` it instantiates a quantum LSTM that uses a small PQC for each gate, otherwise it falls back to a classical LSTM.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# 1. Unified estimator
# --------------------------------------------------------------------------- #
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of scalars into a 2‑D tensor with a batch dimension."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class UnifiedBaseEstimator:
    """Hybrid estimator that works with both PyTorch models and Qiskit circuits.

    Parameters
    ----------
    model_or_circuit
        Either a :class:`torch.nn.Module` (classical) or a Qiskit
        :class:`qiskit.circuit.QuantumCircuit` (quantum).
    """

    def __init__(self, model_or_circuit: Union[nn.Module, Any]) -> None:
        self._model = model_or_circuit
        # Detect if we are dealing with a Qiskit circuit
        self._is_qiskit = (
            hasattr(model_or_circuit, "parameters")
            and hasattr(model_or_circuit, "assign_parameters")
        )
        self._params: List[Any] | None = None
        if self._is_qiskit:
            self._params = list(model_or_circuit.parameters)

    def _evaluate_pytorch(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self._model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self._model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

    def _evaluate_qiskit(
        self,
        observables: Iterable[Any],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        # Import optional dependencies lazily
        from qiskit import Aer, execute
        from qiskit.quantum_info import Statevector

        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self._model.assign_parameters(
                dict(zip(self._params, params)), inplace=False
            )
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate(
        self,
        observables: Iterable[Any],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float | complex]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of observable callables (PyTorch) or Qiskit operators.
        parameter_sets
            Sequence of parameter tuples to bind to the model/circuit.
        shots
            If provided, Gaussian shot noise is added to the deterministic
            expectation values. The standard deviation is ``1/sqrt(shots)``.
        seed
            Random seed used for shot noise.
        """
        if self._is_qiskit:
            raw = self._evaluate_qiskit(observables, parameter_sets)
        else:
            raw = self._evaluate_pytorch(observables, parameter_sets)

        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noisy: List[List[float | complex]] = []
        for row in raw:
            noisy_row = [
                float(rng.normal(float(val), max(1e-6, 1 / shots))) for val in row
            ]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
# 2. Unified LSTMTagger
# --------------------------------------------------------------------------- #
class UnifiedLSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM.

    Parameters
    ----------
    embedding_dim
    hidden_dim
    vocab_size
    tagset_size
    n_qubits
        If ``>0`` a quantum LSTM is instantiated via the :mod:`UnifiedQML` module.
        Otherwise a standard :class:`torch.nn.LSTM` is used.
    quantum_module_path
        Optional import path for the quantum module (defaults to ``UnifiedQML``).
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        quantum_module_path: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            mod_path = quantum_module_path or "UnifiedQML"
            try:
                qm = __import__(mod_path, fromlist=["QLSTMQuantum"])
                QLSTMQuantum = getattr(qm, "QLSTMQuantum")
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    f"Failed to import quantum module '{mod_path}'. "
                    f"Ensure it is available on sys.path."
                ) from exc
            self.lstm = QLSTMQuantum(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        # LSTM expects (seq_len, batch, input_size)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.nn.functional.log_softmax(tag_logits, dim=1)


__all__ = ["UnifiedBaseEstimator", "UnifiedLSTMTagger"]
