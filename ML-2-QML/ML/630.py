import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

class EstimatorQNN(nn.Module):
    """
    Flexible feed‑forward regressor that generalises the original 2‑→8‑→4‑→1 network.
    Parameters
    ----------
    input_dim : int, default 2
        Number of input features.
    hidden_dims : list[int], default [8, 4]
        Sizes of hidden layers.
    output_dim : int, default 1
        Size of the output layer.
    activation : nn.Module, default nn.Tanh
        Non‑linearity applied after each hidden layer.
    dropout : float, default 0.0
        Dropout rate; 0 means no dropout.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: list[int] | tuple[int,...] = (8, 4),
                 output_dim: int = 1,
                 activation: nn.Module = nn.Tanh,
                 dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(self,
            X,
            y,
            epochs: int = 200,
            lr: float = 1e-3,
            batch_size: int = 64,
            verbose: bool = False) -> None:
        """
        Train the network.
        Parameters
        ----------
        X : array‑like, shape (n_samples, n_features)
        y : array‑like, shape (n_samples, )
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self.forward(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataset)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.6f}")

    def predict(self, X):
        """
        Predict on new data.
        Parameters
        ----------
        X : array‑like, shape (n_samples, n_features)
        Returns
        -------
        preds : np.ndarray, shape (n_samples, )
        """
        self.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            return self.forward(X).cpu().numpy().squeeze()

__all__ = ["EstimatorQNN"]
