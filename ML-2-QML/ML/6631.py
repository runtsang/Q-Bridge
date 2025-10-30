import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

class FCL(nn.Module):
    """
    Fully connected layer that supports batch inference, optional noise and training.
    The original seed's run method is preserved for compatibility.
    """
    def __init__(self, n_features: int = 1, noise_std: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = torch.tanh(out)
        if self.noise_std > 0.0:
            noise = torch.randn_like(out) * self.noise_std
            out = out + noise
        return out

    def run(self, thetas: list[float]) -> torch.Tensor:
        """
        Mimics the original seed's run method.
        Accepts a list of input scalars and returns the layer output.
        """
        values = torch.tensor(thetas, dtype=torch.float32).unsqueeze(1)
        return self.forward(values).detach()

    def train_model(self, X: torch.Tensor, y: torch.Tensor,
                    epochs: int = 100, lr: float = 1e-3, batch_size: int = 32):
        """
        Simple training loop using MSE loss.
        """
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

__all__ = ["FCL"]
