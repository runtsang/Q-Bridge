"""Classical implementation of a hybrid LSTM with optional feed‑forward refinement."""
from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedQLSTM(nn.Module):
    """
    Classical LSTM cell where each gate optionally passes through
    an Estimator‑style feed‑forward network.  The network mirrors
    the simple regression architecture from EstimatorQNN and can
    be toggled per gate via the ``use_ff`` flag.
    """
    def __init__(self, input_dim: int, hidden_dim: int, use_ff: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_ff = use_ff

        # Linear projections for gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Optional Estimator‑style feed‑forward refinement
        if self.use_ff:
            self.ff = nn.Sequential(
                nn.Linear(hidden_dim, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, hidden_dim),
            )

    def _apply_ff(self, gate: torch.Tensor) -> torch.Tensor:
        if self.use_ff:
            return self.ff(gate)
        return gate

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self._apply_ff(self.forget_linear(combined)))
            i = torch.sigmoid(self._apply_ff(self.input_linear(combined)))
            g = torch.tanh(self._apply_ff(self.update_linear(combined)))
            o = torch.sigmoid(self._apply_ff(self.output_linear(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
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
    """
    Sequence tagging model that can switch between the hybrid
    UnifiedQLSTM and a vanilla nn.LSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        use_ff: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = UnifiedQLSTM(embedding_dim, hidden_dim, use_ff=use_ff)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["UnifiedQLSTM", "LSTMTagger"]
