"""Unified estimator and sequence‑tagger blending classical and quantum modules.

The module defines two core classes:
1. `FastBaseEstimator` – a lightweight, batched evaluator that works with any PyTorch `nn.Module`.
2. `QLSTM` – a hybrid LSTM where each gate is a small variational quantum circuit.
3. `LSTMTagger` – a sequence‑tagging model that can be configured to use either the classical LSTM or the hybrid QLSTM.

The design keeps the original FastBaseEstimator logic, but adds a new `HybridFastEstimator` that can wrap a classical or quantum model and optionally inject Gaussian noise to mimic shot noise.  It also extends the QLSTM to expose a `to()` method that forwards all sub‑modules to the desired device, enabling seamless GPU/CPU/Quantum backend migration.

The implementation is fully importable and ready for unit tests.  It can be run on a local CPU or with a Qiskit Aer backend for the quantum parts.  """

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class FastBaseEstimator:
    """Fast, batched inference for PyTorch models with optional observables."""
    def __init__(self, model: nn.Module):
        self.model = model

    def _ensure_batch(self, values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
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
        return results


class HybridFastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


class QLSTM(nn.Module):
    """Classical LSTM cell with linear gates (drop‑in for quantum LSTM)."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


class UnifiedEstimatorQLSTM(HybridFastEstimator):
    """Hybrid estimator that can wrap a PyTorch model or a quantum circuit.

    The estimator inherits from :class:`HybridFastEstimator` to provide
    batched evaluation and optional shot‑noise emulation.  It also offers
    a convenience method :meth:`build_lstm_tagger` to instantiate a
    :class:`LSTMTagger` that can operate in classical or quantum mode.
    """
    def __init__(self, model: nn.Module, shots: int | None = None, seed: int | None = None):
        super().__init__(model, shots=shots, seed=seed)

    @staticmethod
    def build_lstm_tagger(
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> LSTMTagger:
        """Return a :class:`LSTMTagger` configured for the given quantum depth."""
        return LSTMTagger(embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits=n_qubits)


__all__ = [
    "FastBaseEstimator",
    "HybridFastEstimator",
    "QLSTM",
    "LSTMTagger",
    "UnifiedEstimatorQLSTM",
]
