import torch
import torch.nn as nn
from torch.optim import Adam

class FCL(nn.Module):
    """
    Extended fully connected neural network with customizable depth and dropout.
    """
    def __init__(self, n_features: int = 1, hidden_sizes: list[int] | None = None,
                 dropout: float = 0.0, lr: float = 1e-3):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 16]
        layers = []
        in_f = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_f, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_f = h
        layers.append(nn.Linear(in_f, 1))
        self.network = nn.Sequential(*layers)
        self.optimizer = Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def run(self, thetas: torch.Tensor | list[float]) -> torch.Tensor:
        """
        Evaluate the network on a batch of inputs.
        Thetas should be a 1‑D iterable or tensor of shape (batch, n_features).
        """
        if isinstance(thetas, (list, tuple)):
            thetas = torch.tensor(thetas, dtype=torch.float32)
        if thetas.ndim == 1:
            thetas = thetas.unsqueeze(0)
        return self.forward(thetas)

    def train_step(self, thetas, targets, loss_fn=nn.MSELoss()):
        self.train()
        self.optimizer.zero_grad()
        preds = self.forward(thetas)
        loss = loss_fn(preds, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

__all__ = ["FCL"]
