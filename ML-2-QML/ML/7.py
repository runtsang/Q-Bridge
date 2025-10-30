"""Enhanced classical sampler network with residual connections and training utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class SamplerQNN(nn.Module):
    """A more expressive sampler network with two hidden layers, batch norm, dropout, and a sample method."""
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] = [64, 64], output_dim: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        return F.softmax(self.net(inputs), dim=-1)

    def sample(self, batch_size: int = 1, device: str | None = None) -> torch.Tensor:
        """Generate samples from the learned distribution."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        with torch.no_grad():
            logits = torch.randn(batch_size, 2, device=device)
            probs = self.forward(logits)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def fit(self, data_loader, epochs: int = 10, lr: float = 1e-3, device: str | None = None):
        """Simple training loop to fit the sampler to target probabilities."""
        self.train()
        if device is None:
            device = torch.device("cpu")
        self.to(device)
        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.KLDivLoss(reduction="batchmean")
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                logits = self.forward(inputs)
                loss = criterion(torch.log(logits + 1e-8), targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            # print(f"Epoch {epoch+1}/{epochs} loss: {epoch_loss/len(data_loader):.4f}")

__all__ = ["SamplerQNN"]
