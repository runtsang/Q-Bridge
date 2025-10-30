import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable

class FCL(nn.Module):
    """
    Extended fully connected layer implemented as a small feed‑forward network.
    Supports dropout, multiple hidden layers and a simple training API.
    """

    def __init__(self,
                 input_dim: int = 1,
                 hidden_dims: Iterable[int] = (32, 16),
                 output_dim: int = 1,
                 dropout: float = 0.1,
                 device: str | None = None) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 100,
            lr: float = 1e-3,
            batch_size: int = 32,
            verbose: bool = False) -> None:
        """
        Train the network using MSE loss and Adam optimizer.
        """
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1).to(self.device)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.forward(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss/len(dataset):.4f}")

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Forward pass on a batch of input samples.
        ``thetas`` is expected to be a 2‑D array of shape (batch, input_dim).
        """
        self.eval()
        with torch.no_grad():
            inp = torch.from_numpy(thetas.astype(np.float32)).to(self.device)
            out = self.forward(inp)
        return out.cpu().numpy().squeeze()
