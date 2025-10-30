"""Hybrid classical LSTM implementation with optional noise estimator.

This module implements a drop‑in replacement for the original QLSTM
class.  The gates are realised by fully‑connected linear layers
followed by the standard LSTM non‑linearities.  A lightweight
``FastEstimator`` is provided to add Gaussian shot noise to the
deterministic outputs, mirroring the behaviour of the QML variant
but remaining purely classical.

The API is compatible with the original `QLSTM` and `LSTMTagger`
classes, so existing training scripts can be reused without
modification.
"""

from __future__ import annotations

from typing import Tuple, Iterable, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Iterable[float]) -> torch.Tensor:
    t = torch.as_tensor(list(values), dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


class QLSTM(nn.Module):
    """Classical LSTM cell with linear gates.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input at each time step.
    hidden_dim : int
        Dimensionality of the hidden state.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class LSTMTagger(nn.Module):
    """Tagging model that uses either the classical `QLSTM` or a native
    `nn.LSTM` layer.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


class FastBaseEstimator:
    """Evaluate a PyTorch model for a batch of parameter sets.

    The model is expected to accept a tensor of shape
    ``(batch, input_dim)`` and return a tensor of arbitrary shape.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Iterable[Iterable[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
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


class FastEstimator(FastBaseEstimator):
    """Wraps :class:`FastBaseEstimator` and injects Gaussian shot noise.

    Parameters
    ----------
    shots : int | None
        If ``None`` the estimator is deterministic.
    seed : int | None
        Random seed for reproducibility.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Iterable[Iterable[float]],
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
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["QLSTM", "LSTMTagger", "FastBaseEstimator", "FastEstimator"]
