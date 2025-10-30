"""Enhanced regression estimator with configurable depth and regularization."""

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

class EstimatorQNN(nn.Module):
    """A flexible feed‑forward regressor supporting residual blocks, dropout, and batch‑norm."""
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | None = None,
        output_dim: int = 1,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 4]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Train the network with Adam and optional early stopping."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        best_loss = float("inf")
        patience, counter = 10, 0
        best_state = None
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self(xb).squeeze()
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.6f}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                counter = 0
                best_state = self.state_dict()
            else:
                counter += 1
                if counter >= patience:
                    if verbose:
                        print("Early stopping")
                    break
        if best_state is not None:
            self.load_state_dict(best_state)

    def predict(self, X: torch.Tensor, device: str | None = None) -> torch.Tensor:
        """Return predictions for X."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        with torch.no_grad():
            X = X.to(device)
            return self(X).cpu()

__all__ = ["EstimatorQNN"]
