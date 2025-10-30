"""Hybrid LSTM with quantum‑inspired estimator gating.

The model extends a classical LSTM cell by wrapping each gate with a
small neural estimator that emulates the behaviour of a parameterised
quantum circuit (see EstimatorQNN).  The estimator learns to modulate
the gate activations, providing a learnable, non‑linear scaling that
captures quantum‑like correlations without requiring a quantum backend.

Classes
-------
CombinedQLSTM
    Hybrid LSTM cell that can operate in three modes:
        * classical (default) – simple linear gates;
        * quantum‑inspired – gate activations are multiplied by the
          output of a small estimator network;
        * fully quantum – delegated to a separate QML module
          (importable as ``qlm.CombinedQLSTM``).
LSTMTagger
    Wrapper that applies the cell to a sequence and projects to tags.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _HybridEstimator(nn.Module):
    """Small feed‑forward network that mimics a parameterised quantum circuit."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.Sigmoid(),  # keep output in (0,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CombinedQLSTM(nn.Module):
    """Hybrid LSTM cell that optionally augments gates with a quantum‑inspired estimator."""
    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = 0,
                 use_estimator: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_estimator = use_estimator

        # Classical linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if self.use_estimator:
            # One estimator per gate
            self.forget_est = _HybridEstimator(hidden_dim)
            self.input_est = _HybridEstimator(hidden_dim)
            self.update_est = _HybridEstimator(hidden_dim)
            self.output_est = _HybridEstimator(hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Classical gate outputs
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            if self.use_estimator:
                # Estimator outputs act as multiplicative scalars
                f = f * self.forget_est(f)
                i = i * self.input_est(i)
                g = g * self.update_est(g)
                o = o * self.output_est(o)

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
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class LSTMTagger(nn.Module):
    """Sequence tagging model that can employ a hybrid LSTM cell."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_estimator: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if use_estimator:
            self.lstm = CombinedQLSTM(
                embedding_dim, hidden_dim, n_qubits=n_qubits, use_estimator=True
            )
        else:
            self.lstm = CombinedQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["CombinedQLSTM", "LSTMTagger"]
