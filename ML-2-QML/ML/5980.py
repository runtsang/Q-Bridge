import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    A deeper, regularised sampler network.

    Extends the original two‑layer architecture with batch‑norm,
    ReLU activations, dropout and an optional custom initialisation.
    The final layer uses a softmax to produce a probability vector
    over the two output classes, mirroring the quantum sampler.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 output_dim: int = 2,
                 dropout: float = 0.1,
                 init: str = "kaiming_uniform_") -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
        )

        if init == "kaiming_uniform_":
            nn.init.kaiming_uniform_(self.net[0].weight)
            nn.init.kaiming_uniform_(self.net[3].weight)
            nn.init.kaiming_uniform_(self.net[6].weight)
        elif init == "xavier_uniform_":
            nn.init.xavier_uniform_(self.net[0].weight)
            nn.init.xavier_uniform_(self.net[3].weight)
            nn.init.xavier_uniform_(self.net[6].weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNN"]
