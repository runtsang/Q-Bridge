import torch
from torch import nn
from torch.nn import functional as F

class EstimatorNN(nn.Module):
    """
    Flexible fully‑connected regressor.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input features.
    hidden_layers : tuple[int,...], default (8, 4)
        Sizes of hidden layers. If empty, the network is a single linear layer.
    output_dim : int, default 1
        Size of the output layer.
    activation : str, default "tanh"
        Activation function to use between layers.
    dropout : float, default 0.0
        Dropout probability applied after each hidden layer.
    l2 : float, default 0.0
        L2 regularisation coefficient applied to all linear weights.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_layers: tuple[int,...] = (8, 4),
        output_dim: int = 1,
        activation: str = "tanh",
        dropout: float = 0.0,
        l2: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        act = getattr(nn, activation.capitalize())()
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), act])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.l2 = l2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def l2_loss(self) -> torch.Tensor:
        """Return L2 penalty for all linear layers."""
        if self.l2 == 0.0:
            return torch.tensor(0.0, device=self.net[0].weight.device)
        return self.l2 * sum(
            torch.norm(m.weight, 2) ** 2
            for m in self.modules()
            if isinstance(m, nn.Linear)
        )

def EstimatorQNN() -> EstimatorNN:
    """Convenience constructor matching the original API."""
    return EstimatorNN()

__all__ = ["EstimatorQNN", "EstimatorNN"]
