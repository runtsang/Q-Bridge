import torch
from torch import nn
import math

class EstimatorQNN(nn.Module):
    """
    Classical regression estimator that maps a 2‑dimensional input to a scalar
    output.  The architecture mirrors the original EstimatorQNN seed but
    incorporates:
        • Batch‑normalisation of the raw input for stable training.
        • A two‑layer MLP with Tanh activations and a residual connection
          from the first hidden layer to the second.
        • L2 weight‑decay can be applied by passing a `weight_decay`
          argument to the optimiser; the class stores the value for
          reference.
        • Optional output scaling to match target ranges.
    """
    def __init__(self,
                 in_features: int = 2,
                 hidden_sizes: tuple[int, int] = (8, 4),
                 out_features: int = 1,
                 weight_decay: float = 0.0,
                 scale_input: bool = True,
                 output_scale: float | None = None):
        super().__init__()
        self.scale_input = scale_input
        self.output_scale = output_scale
        self.weight_decay = weight_decay

        if self.scale_input:
            self.bn = nn.BatchNorm1d(in_features)

        self.linear1 = nn.Linear(in_features, hidden_sizes[0])
        self.linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.linear3 = nn.Linear(hidden_sizes[1], out_features)

        # initialise weights for reproducibility
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.zeros_(self.linear3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: input → BN → linear1 → tanh → linear2 → add skip
        → tanh → linear3 → optional scaling.
        """
        if self.scale_input:
            x = self.bn(x)
        h1 = torch.tanh(self.linear1(x))
        h2 = torch.tanh(self.linear2(h1) + h1)  # skip connection
        out = self.linear3(h2)
        if self.output_scale is not None:
            out = out * self.output_scale
        return out

__all__ = ["EstimatorQNN"]
