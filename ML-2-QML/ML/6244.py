import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Iterable, List

class FCL(nn.Module):
    """
    A flexible fully‑connected neural network that emulates the behaviour
    of the original toy example while offering a richer interface for
    experimentation.  The network supports an arbitrary number of hidden
    layers, dropout, and L1 regularisation.  A ``run`` method is kept for
    backward compatibility; it returns the mean activation of the final
    layer.
    """
    def __init__(self,
                 input_dim: int = 1,
                 hidden_dims: List[int] | None = None,
                 dropout: float = 0.0,
                 l1_lambda: float = 0.0,
                 device: str | torch.device = "cpu") -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32]
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self.l1_lambda = l1_lambda
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Backwards‑compatible wrapper that mimics the original ``run`` API.
        ``thetas`` are interpreted as a 1‑D sequence of input values that
        are passed through the network and the mean of the final layer
        activations is returned as a NumPy array.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32, device=self.device).view(-1, 1)
        with torch.no_grad():
            out = self.forward(values).mean(dim=0)
        return out.cpu().detach().numpy()

    def l1_regulariser(self) -> torch.Tensor:
        """L1 penalty over all learnable parameters."""
        if self.l1_lambda == 0.0:
            return torch.tensor(0.0, device=self.device)
        return self.l1_lambda * torch.tensor(
            sum(p.abs().sum() for p in self.parameters()), device=self.device
        )

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean‑squared‑error plus optional L1 regularisation."""
        mse = F.mse_loss(predictions, targets)
        return mse + self.l1_regulariser()

    def fit(self,
            X: torch.Tensor,
            y: torch.Tensor,
            epochs: int = 200,
            lr: float = 1e-3,
            batch_size: int = 32,
            verbose: bool = False) -> List[torch.Tensor]:
        """
        Simple training loop that optimises the network for the supplied
        ``X`` and ``y``.  Returns the history of the loss values.
        """
        X, y = X.to(self.device), y.to(self.device)
        optimizer = Adam(self.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.forward(xb)
                loss = self.loss(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.6f}")

        return history
