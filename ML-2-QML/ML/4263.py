"""Hybrid classical LSTM tagger with quantum‑style sampler and regression head.

The module is drop‑in compatible with the original ``QLSTM`` tagger interface.
A 2→4→2 softmax encoder mimics a quantum measurement and expands the
embedding dimension by 2.  A regression head is added for auxiliary
sequence‑level regression tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SamplerModule(nn.Module):
    """Simple 2→4→2 softmax encoder mimicking a quantum sampler."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


class QLSTM(nn.Module):
    """Classical LSTM cell with linear gates."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits  # unused in the classical branch
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class HybridQLSTM(nn.Module):
    """Drop‑in hybrid LSTM tagger with optional quantum‑style gating."""

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
        self.sampler = _SamplerModule()

        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim + 2, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim + 2, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden2reg = nn.Linear(hidden_dim, 1)

    def forward(self, sentence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        log_probs : Tensor
            Log‑softmax tag probabilities of shape ``(seq_len, tagset_size)``.
        reg : Tensor
            Regression output of shape ``(seq_len,)``.
        """
        embeds = self.word_embeddings(sentence)
        sampled = self.sampler(embeds)
        combined = torch.cat([embeds, sampled], dim=-1)
        lstm_out, _ = self.lstm(combined.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        reg = self.hidden2reg(lstm_out.view(len(sentence), -1)).squeeze(-1)
        return F.log_softmax(tag_logits, dim=1), reg

    @property
    def regression_head(self) -> nn.Linear:
        """Expose the regression head for external use."""
        return self.hidden2reg


__all__ = ["HybridQLSTM"]
