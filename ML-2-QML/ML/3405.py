"""Hybrid classical LSTMTagger with random Fourier feature augmentation.

This module extends the original QLSTM implementation by adding a
classical random feature layer that mimics the quantum feature map
used in the quantum version. The new class is drop‑in compatible
with the legacy API and can be used in any existing training loop.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RandomFourierFeature(nn.Module):
    """Classical random feature mapping that approximates a quantum
    feature map using sinusoidal activations.
    """

    def __init__(self, input_dim: int, n_features: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.register_buffer("omega", torch.randn(input_dim, n_features))
        self.register_buffer("beta", 2 * math.pi * torch.rand(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.omega
        return torch.cat([torch.sin(proj + self.beta), torch.cos(proj + self.beta)], dim=-1)


class ClassicalQLSTM(nn.Module):
    """Standard LSTM cell implemented with linear gates."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

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


class HybridQLSTMTagger(nn.Module):
    """Sequence tagging model that augments embeddings with a classical
    random feature map before feeding them into an LSTM.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Random feature augmentation of embeddings
        self.embed_augment = RandomFourierFeature(embedding_dim, n_qubits)

        self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed_dim)

        # Apply random feature mapping to each embedding
        seq_len, batch, _ = embeds.shape
        flat = embeds.reshape(-1, self.embedding_dim)
        rf = self.embed_augment(flat)
        rf = rf.reshape(seq_len, batch, -1)

        lstm_out, _ = self.lstm(rf.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTMTagger"]
