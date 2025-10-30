import torch
from torch import nn

class EstimatorQNNHybrid(nn.Module):
    """
    A hybrid regressor that extends the original EstimatorQNN by adding
    multiple hidden layers, batch‑normalisation, ReLU activations and
    dropout.  The architecture is fully parameterisable so that it can
    be tuned for larger datasets or more complex regression tasks.
    """
    def __init__(self, input_dim: int = 2,
                 hidden_dims: list[int] | tuple[int,...] = (32, 16),
                 dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def train_model(self, train_loader, val_loader=None,
                    epochs: int = 50, lr: float = 1e-3) -> None:
        """
        Simple training loop that optimises the network with Adam and
        MSE loss.  Validation loss is printed each epoch if a validation
        loader is supplied.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            self.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = self(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
            if val_loader:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        preds = self(xb)
                        val_loss += criterion(preds, yb).item()
                val_loss /= len(val_loader)
                print(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for inference."""
        self.eval()
        with torch.no_grad():
            return self(x)

__all__ = ["EstimatorQNNHybrid"]
