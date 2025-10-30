"""Hybrid LSTM combining classical linear gates with a quantum‑style sampler for the forget gate."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerModule(nn.Module):
    """Simple 2‑output softmax module that mimics the SamplerQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs expected shape (..., 2)
        return F.softmax(self.net(inputs), dim=-1)


class HybridQLSTM(nn.Module):
    """Drop‑in replacement that uses a classical linear transformer for all gates
    except the forget gate, which is sampled from a 2‑output quantum‑style sampler."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, 2)  # two logits → sampler
        self.input_linear   = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear  = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear  = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum‑style sampler for forget gate
        self.forget_sampler = SamplerModule()

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            # Forget gate via sampler
            f_logits = self.forget_linear(combined)          # (..., 2)
            f_probs  = self.forget_sampler(f_logits)         # (..., 2)
            f = f_probs[:, 0]                                # use first prob
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


class LSTMTagger(nn.Module):
    """Sequence tagging model that can use either :class:`HybridQLSTM` or
    the standard :class:`torch.nn.LSTM`."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim) if use_quantum else nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "LSTMTagger"]
