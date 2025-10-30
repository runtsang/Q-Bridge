"""Enhanced QCNN model with residual connections, dropout, and optional self‑attention.

This class extends the original QCNNModel by adding residual connections every two layers,
dropout for regularisation, and an optional self‑attention block that can be toggled
via the `use_attention` flag.  The architecture remains fully classical and
fully differentiable, making it suitable for integration into standard PyTorch pipelines.
"""

import torch
from torch import nn

class QCNNGen375(nn.Module):
    """Convolution‑inspired feed‑forward network with residuals and attention."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: list[int] | tuple[int,...] = (16, 16, 12, 8, 4, 4),
        dropout: float = 0.2,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.layers = nn.ModuleList()
        self.residuals = nn.ModuleList()
        prev_dim = input_dim

        for i, dim in enumerate(hidden_dims):
            # Main linear + activation block
            self.layers.append(nn.Sequential(nn.Linear(prev_dim, dim), nn.Tanh()))
            # Residual mapping from previous layer to current layer
            if i % 2 == 1:
                self.residuals.append(nn.Linear(hidden_dims[i - 1], dim))
            prev_dim = dim

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(prev_dim, 1)

        if use_attention:
            # Self‑attention over the final feature vector
            self.attn = nn.MultiheadAttention(
                embed_dim=prev_dim, num_heads=2, batch_first=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            out = layer(x)
            if i % 2 == 1:
                # Add residual connection from the preceding block
                res = self.residuals[i // 2](x)
                out = out + res
            out = self.dropout(out)
            x = out

        if self.use_attention:
            # Reshape for the attention module: (batch, seq_len=1, feature)
            x = x.unsqueeze(1)
            x, _ = self.attn(x, x, x)
            x = x.squeeze(1)

        logits = torch.sigmoid(self.head(x))
        return logits

def build_QCNNGen375() -> QCNNGen375:
    """Factory returning the configured QCNNGen375 model."""
    return QCNNGen375()

__all__ = ["QCNNGen375"]
