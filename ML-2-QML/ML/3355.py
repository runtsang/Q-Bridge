import torch
from torch import nn
import torch.nn.functional as F

class EstimatorQNN(nn.Module):
    """
    Deep residual regression network with optional quantum head.
    Combines the lightweight FC model from EstimatorQNN with
    residual blocks, batch‑norm, dropout, and an optional
    callable quantum head that can be injected during construction.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.2,
        quantum_head: callable | None = None,
    ) -> None:
        super().__init__()
        self.quantum_head = quantum_head

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.input_layer(x))
        for block in self.res_blocks:
            residual = out
            out = block(out)
            out = out + residual
        out = self.dropout(out)
        out = self.output_layer(out)

        if self.quantum_head is not None:
            # quantum_head should take a 1‑D tensor of shape (batch,)
            # and return a tensor of the same shape.
            q_out = self.quantum_head(out.squeeze(-1))
            out = out + q_out.unsqueeze(-1)
        return out

__all__ = ["EstimatorQNN"]
