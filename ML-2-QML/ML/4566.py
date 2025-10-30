"""Hybrid classical LSTM module that optionally simulates quantum gates and includes
a fully‑connected head inspired by the Quantum‑NAT architecture.  The class
mirrors the interface of the original QLSTM but adds a flag to toggle between
pure classical gates or quantum‑style linear projections.  A lightweight
FastEstimator wrapper is provided for rapid batch evaluation and optional
shot‑noise simulation.
"""

from __future__ import annotations

from typing import Iterable, Callable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HybridQLSTM(nn.Module):
    """
    Drop‑in replacement for QLSTM that can operate in a pure classical mode
    or with quantum‑style gate simulations.  When ``n_qubits`` is set to a
    positive integer the gates are implemented as linear projections onto a
    small quantum register followed by a sigmoid/tanh activation; otherwise
    the gates are standard PyTorch linear layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_fc: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, input_dim)

        self.is_quantum_style = n_qubits > 0

        if self.is_quantum_style:
            # Quantum‑style gates: linear projection followed by a non‑linear
            self.forget = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim), nn.Sigmoid())
            self.input = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim), nn.Sigmoid())
            self.update = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim), nn.Tanh())
            self.output = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim), nn.Sigmoid())
        else:
            # Classical LSTM gates
            self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.use_fc = use_fc
        if use_fc:
            # Classical fully‑connected head inspired by QFCModel
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
                nn.BatchNorm1d(4),
            )
        else:
            self.fc = None

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        hx, cx = self._init_states(embeds)
        outputs = []
        for x in embeds.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            if self.is_quantum_style:
                f = self.forget(combined)
                i = self.input(combined)
                g = self.update(combined)
                o = self.output(combined)
            else:
                f = torch.sigmoid(self.forget(combined))
                i = torch.sigmoid(self.input(combined))
                g = torch.tanh(self.update(combined))
                o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        lstm_out = torch.cat(outputs, dim=0)
        tag_logits = self.hidden2tag(lstm_out)
        result = F.log_softmax(tag_logits, dim=1)
        if self.use_fc:
            aux = self.fc(lstm_out[-1].unsqueeze(0))
            result = torch.cat([result, aux], dim=1)
        return result


class FastEstimator:
    """
    Lightweight estimator that evaluates the model on a batch of parameter
    sets and observables.  It wraps a PyTorch ``nn.Module`` and optionally
    adds Gaussian shot noise to simulate quantum measurement statistics.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
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
                    value = obs(outputs)
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


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


__all__ = ["HybridQLSTM", "FastEstimator"]
