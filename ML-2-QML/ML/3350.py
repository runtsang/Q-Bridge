'Hybrid estimator with classical fast evaluation and sequence tagging.\n\nThis module implements a lightweight `HybridEstimator` that can evaluate\n a PyTorch model for many parameter sets (optionally with Gaussian shot\n noise) and perform sequence tagging with a standard `nn.LSTM`.  The\n class exposes a unified API that mirrors the quantum version, making\n inter‑comparison straightforward.\n'

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    'Evaluate a PyTorch model for many parameter sets.'
    def __init__(self, model: nn.Module) -> None:
        self.model = model

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
        return results


class FastEstimator(FastBaseEstimator):
    'Adds optional Gaussian shot noise to the deterministic estimator.'
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


class LSTMTagger(nn.Module):
    'Standard sequence tagging model that uses an `nn.LSTM`.'
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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


class HybridEstimator:
    'Unified interface for classical fast evaluation and sequence tagging.'
    def __init__(self, model: nn.Module | None = None, lstm: nn.Module | None = None) -> None:
        self.model = model
        self.lstm = lstm
        if self.model is not None:
            self._estimator = FastEstimator(self.model)
        else:
            self._estimator = None

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        if self._estimator is None:
            raise RuntimeError('No model supplied for evaluation.')
        return self._estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

    def tag(self, sentence: torch.Tensor) -> torch.Tensor:
        if self.lstm is None:
            raise RuntimeError('No LSTM supplied for tagging.')
        return self.lstm(sentence)


__all__ = ['FastBaseEstimator', 'FastEstimator', 'LSTMTagger', 'HybridEstimator']
