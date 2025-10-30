import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionQLSTM(nn.Module):
    """
    Classical hybrid self‑attention + LSTM module.
    Combines a dense self‑attention head with an LSTM encoder.
    The architecture is fully differentiable and can be used as a drop‑in replacement
    for either a pure attention or a pure LSTM block.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, n_qubits: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Classical self‑attention layers
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        # Classical LSTM encoder
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=False)

        # Projection back to embedding space
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (seq_len, batch, embed_dim).

        Returns
        -------
        torch.Tensor
            Output sequence of shape (seq_len, batch, embed_dim).
        """
        # Self‑attention
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = F.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        attn_out = scores @ V

        # LSTM encoding
        lstm_out, _ = self.lstm(attn_out)

        # Project back to original dimension
        return self.proj(lstm_out)
