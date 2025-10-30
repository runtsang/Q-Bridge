import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SelfAttention(nn.Module):
    """
    Classical self‑attention with optional LSTM gating and classification head.
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        use_lstm: bool = False,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_lstm = use_lstm

        # Linear projections for query, key and value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Optional LSTM gating
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
            )
            self.lstm_linear = nn.Linear(lstm_hidden, embed_dim)

        # Feed‑forward head for binary classification
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape ``(batch, seq_len, embed_dim)``.

        Returns
        -------
        logits : torch.Tensor
            Log‑probabilities of shape ``(batch, seq_len, 2)``.
        """
        B, T, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim**0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Optional LSTM gating
        if self.use_lstm:
            out_lstm, _ = self.lstm(out)
            out = self.lstm_linear(out_lstm) * out  # element‑wise gating

        logits = self.head(out)
        return logits

__all__ = ["SelfAttention"]
