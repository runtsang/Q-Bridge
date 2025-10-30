import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class EstimatorQNN(nn.Module):
    """
    Robust feed‑forward regression network.
    Features batch‑normalisation, dropout, and an early‑stopping training helper.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: tuple[int,...] = (32, 16),
                 output_dim: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 200,
            batch_size: int = 32,
            lr: float = 1e-3,
            early_stop_patience: int = 20,
            verbose: bool = False) -> None:
        """Train the network on the supplied data."""
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                                torch.tensor(y, dtype=torch.float32).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        best_loss = np.inf
        patience = 0
        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)

            if verbose:
                print(f"Epoch {epoch:03d} – loss: {epoch_loss:.6f}")

            # early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stop_patience:
                    if verbose:
                        print(f"Early stopping after {epoch} epochs")
                    break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for the given inputs."""
        self.eval()
        with torch.no_grad():
            return self(torch.tensor(X, dtype=torch.float32)).numpy().squeeze()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score."""
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))

__all__ = ["EstimatorQNN"]
