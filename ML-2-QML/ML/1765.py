"""Enhanced classical LSTM with dropout, residuals, and multi‑layer support."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

class QLSTM(nn.Module):
    """Drop‑in replacement that extends the original QLSTM to support
    multiple layers, configurable dropout and optional residual connections.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.residual = residual

        # A list of LSTMCell modules – one per layer.
        self.cells = nn.ModuleList(
            [
                nn.LSTMCell(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                )
                for i in range(n_layers)
            ]
        )

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if states is not None:
            h, c = states
            return h, c
        batch_size = inputs.size(0)
        device = inputs.device
        h = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.n_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.n_layers)]
        return h, c

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Run the multi‑layer LSTM over a sequence.

        Args:
            inputs: Tensor of shape (seq_len, batch, input_dim).
            states: Optional tuple of hidden and cell states for each layer.

        Returns:
            output: Tensor of shape (seq_len, batch, hidden_dim) – the output
                of the last layer.
            (h, c): Final hidden and cell states for each layer.
        """
        h, c = self._init_states(inputs, states)
        outputs = []

        for t in range(inputs.size(0)):
            x = inputs[t]
            new_h, new_c = [], []
            for i, cell in enumerate(self.cells):
                h_i, c_i = cell(x, (h[i], c[i]))
                if self.dropout > 0.0:
                    h_i = F.dropout(h_i, p=self.dropout, training=self.training)
                if self.residual and i > 0:
                    h_i = h_i + x  # residual on the input of the layer
                new_h.append(h_i)
                new_c.append(c_i)
                x = h_i  # feed into next layer
            h, c = new_h, new_c
            outputs.append(h[-1].unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (h, c)


class LSTMTagger(nn.Module):
    """Sequence tagging model that can use the upgraded classical QLSTM
    or the baseline ``nn.LSTM``.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        n_layers: int = 1,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_layers=n_layers,
                dropout=dropout,
                residual=residual,
            )
        else:
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=False,
            )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)


__all__ = ["QLSTM", "LSTMTagger"]
