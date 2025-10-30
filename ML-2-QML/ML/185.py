import torch
from torch import nn, optim
import numpy as np

class EstimatorQNNGen204(nn.Module):
    """
    A configurable feed‑forward regressor.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_layers : list[int]
        Sizes of hidden layers. Default [64, 32].
    dropout : float | None
        Dropout probability. If None no dropout is applied.
    activation : nn.Module
        Activation function to use. Default nn.Tanh().
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_layers: list[int] | None = None,
                 dropout: float | None = None,
                 activation: nn.Module | None = None) -> None:
        super().__init__()
        hidden_layers = hidden_layers or [64, 32]
        activation = activation or nn.Tanh()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), activation])
            if dropout:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 200,
            lr: float = 1e-3,
            batch_size: int = 32,
            verbose: bool = True) -> None:
        """
        Train the network using MSE loss and Adam optimiser.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, input_dim).
        y : np.ndarray
            Target values, shape (n_samples,).
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        batch_size : int
            Mini‑batch size.
        verbose : bool
            If True prints epoch loss.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)
            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch:3d}/{epochs} - Loss: {epoch_loss:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained network.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, input_dim).

        Returns
        -------
        np.ndarray
            Predicted values, shape (n_samples,).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            pred = self(X_t).cpu().numpy().squeeze()
        return pred

__all__ = ["EstimatorQNNGen204"]
