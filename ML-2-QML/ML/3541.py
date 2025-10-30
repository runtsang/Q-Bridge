"""Hybrid classical LSTM with kernel‑based gate feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelLayer(nn.Module):
    """Encodes a vector into a kernel feature space using a trainable RBF kernel."""
    def __init__(self, input_dim: int, n_prototypes: int = 32, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)  # (batch, n_proto, dim)
        dist_sq = torch.sum(diff * diff, dim=2)  # (batch, n_proto)
        return torch.exp(-self.gamma * dist_sq)  # (batch, n_proto)


class QLSTM(nn.Module):
    """Classical LSTM where gates are driven by kernel‑encoded features."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_prototypes: int = 32,
        gamma: float = 1.0,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_layer = KernelLayer(input_dim + hidden_dim, n_prototypes, gamma)
        self.forget_linear = nn.Linear(n_prototypes, hidden_dim)
        self.input_linear  = nn.Linear(n_prototypes, hidden_dim)
        self.update_linear = nn.Linear(n_prototypes, hidden_dim)
        self.output_linear = nn.Linear(n_prototypes, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)  # (batch, input+hidden)
            k = self.kernel_layer(combined)       # (batch, n_proto)
            f = torch.sigmoid(self.forget_linear(k))
            i = torch.sigmoid(self.input_linear(k))
            g = torch.tanh(self.update_linear(k))
            o = torch.sigmoid(self.output_linear(k))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the hybrid QLSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_prototypes: int = 32,
        gamma: float = 1.0,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_prototypes=n_prototypes,
                gamma=gamma,
                n_qubits=n_qubits,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
