import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, List, Optional

class FCL(nn.Module):
    """
    An extensible fully‑connected neural layer that can be used as a drop‑in
    replacement for the original toy implementation.

    Parameters
    ----------
    n_features : int
        Size of the input vector.
    hidden_layers : List[int], optional
        Sizes of hidden layers.  If ``None`` a single linear layer is used.
    dropout : float, optional
        Drop‑out probability applied after every hidden layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []

        input_size = n_features
        if hidden_layers is None:
            hidden_layers = []

        for size in hidden_layers:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_size = size

        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.model(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the original API: interpret ``thetas`` as a one‑dimensional
        input vector, run a forward pass and return the mean of the output
        as a NumPy array.
        """
        device = next(self.parameters()).device
        tensor = torch.as_tensor(list(thetas), dtype=torch.float32, device=device).view(-1, 1)
        with torch.no_grad():
            out = self.forward(tensor)
            mean = out.mean(dim=0)
        return mean.detach().cpu().numpy()

    def train_on(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> None:
        """
        Quick training helper that optimises the network on supplied data.
        """
        dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32).unsqueeze(-1),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(X))
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.forward(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
            if verbose:
                print(f"Epoch {epoch+1:03d} | loss: {loss.item():.6f}")
