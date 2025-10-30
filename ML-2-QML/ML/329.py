import numpy as np
import torch
from torch import nn

class FCL(nn.Module):
    """
    Robust classical fully‑connected layer that emulates the behaviour of the
    original quantum fully‑connected layer.  It accepts batch inputs,
    optionally applies dropout, and exposes a `run` method that returns the
    mean activation as a 1‑element numpy array – matching the original API.
    """
    def __init__(self, n_features: int = 1, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Initialise weights to mimic a quantum‑style distribution
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: linear → tanh → dropout.
        """
        return torch.tanh(self.linear(x))

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Mimic the quantum layer by feeding a 1‑D array of parameters,
        computing the forward pass, and returning the mean over the batch.
        The output shape matches the original: (1,).
        """
        x = torch.as_tensor(thetas, dtype=torch.float32).unsqueeze(-1)
        out = self.forward(x)
        mean_val = out.mean().item()
        return np.array([mean_val])

    def train(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 200):
        """
        Very light‑weight training loop using Adam optimizer.
        """
        dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.forward(xb)
                loss = loss_fn(pred, yb.unsqueeze(-1))
                loss.backward()
                optimizer.step()
