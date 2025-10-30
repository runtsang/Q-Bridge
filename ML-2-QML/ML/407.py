import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """
    A configurable feed‑forward regressor that can be tuned with
    hidden layer sizes, activation functions, dropout, and weight
    decay.  It mirrors the original EstimatorQNN but adds a
    *dynamic architecture* and *regularisation* layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int,...] = (16, 8),
        output_dim: int = 1,
        activation: nn.Module = nn.Tanh,
        dropout_prob: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(dropout_prob))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def l2_regularization(self) -> torch.Tensor:
        """Return the L2 penalty over all parameters."""
        l2 = torch.tensor(0.0, device=self.parameters().__next__().device)
        for p in self.parameters():
            l2 = l2 + torch.norm(p, 2) ** 2
        return l2 * self.weight_decay

__all__ = ["EstimatorQNN"]
