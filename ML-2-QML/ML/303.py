"""Hybrid classical LSTM with residual skip-connection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTM(nn.Module):
    """Classical LSTM cell with a residual skip‑connection from the previous hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.total_dim = input_dim + hidden_dim
        self.forget = nn.Linear(self.total_dim, hidden_dim, bias=bias)
        self.input  = nn.Linear(self.total_dim, hidden_dim, bias=bias)
        self.update = nn.Linear(self.total_dim, hidden_dim, bias=bias)
        self.output = nn.Linear(self.total_dim, hidden_dim, bias=bias)
        self.res_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(seq_len, batch, input_dim)``.
        states : tuple, optional
            Initial hidden and cell states of shape ``(batch, hidden_dim)``.
            If ``None`` zero‑states are used.

        Returns
        -------
        outputs : torch.Tensor
            Tensor of shape ``(seq_len, batch, hidden_dim)``.
        final_state : tuple
            Final hidden and cell states.
        """
        if states is None:
            batch_size = inputs.size(1)
            device = inputs.device
            h = torch.zeros(batch_size, self.hidden_dim, device=device)
            c = torch.zeros(batch_size, self.hidden_dim, device=device)
        else:
            h, c = states

        outputs = []
        for x in inputs.unbind(dim=0):
            h_prev = h
            combined = torch.cat([x, h_prev], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))

            c = f * c + i * g
            h = o * torch.tanh(c)

            # Residual skip‑connection
            h = h + self.res_proj(h_prev)

            outputs.append(h.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (h, c)

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the residual classical LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, bias=bias)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTM", "LSTMTagger"]
