"""
Enhanced classical LSTM module with optional residual, attention, and dropout.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLSTM(nn.Module):
    """
    Classical LSTM cell with optional residual connections, multi‑head attention,
    and dropout. The interface matches the original seed so it can be swapped
    in the same downstream code.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        use_residual: bool = False,
        n_heads: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # Standard LSTM gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Optional self‑attention on the input sequence
        if n_heads > 1:
            self.attention = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=n_heads, batch_first=True
            )
        else:
            self.attention = None

        self.use_residual = use_residual

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Sequence of shape (seq_len, batch, input_dim).
        states : Tuple[torch.Tensor, torch.Tensor] | None
            Optional initial hidden and cell states.
        """
        hx, cx = self._init_states(inputs, states)

        outputs = []
        seq_len = inputs.size(0)

        # Optionally compute self‑attention over the whole sequence
        if self.attention is not None:
            # MultiheadAttention expects (batch, seq, embed)
            attn_out, _ = self.attention(
                inputs.transpose(0, 1), inputs.transpose(0, 1), inputs.transpose(0, 1)
            )
            attn_out = attn_out.transpose(0, 1)  # back to (seq, batch, embed)
            # Merge attention with original input
            inputs = inputs + attn_out

        for t in range(seq_len):
            x = inputs[t]
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            if self.use_residual:
                hx = hx + x  # residual connection to the input at this timestep

            outputs.append(self.dropout(hx.unsqueeze(0)))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

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
    Sequence tagging model that can use either the enhanced QLSTM or PyTorch's
    built‑in LSTM. The constructor mimics the original signature for
    drop‑in compatibility.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dropout: float = 0.0,
        use_residual: bool = False,
        n_heads: int = 1,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if use_quantum:
            # Quantum implementation is defined in the QML module
            from.QLSTM__gen289 import QLSTM as QuantumQLSTM  # type: ignore
            self.lstm = QuantumQLSTM(
                embedding_dim, hidden_dim, n_qubits=n_qubits
            )
        else:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                dropout=dropout,
                use_residual=use_residual,
                n_heads=n_heads,
            )

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
