import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class QCNNHybrid(nn.Module):
    """
    Classical convolution-inspired network that mirrors the quantum convolution steps.
    Adds a BatchNorm layer, flexible hidden layers, and a train method.
    """
    def __init__(self, input_dim: int = 8, hidden_sizes: list[int] = None, output_dim: int = 1):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [16, 12, 8, 4]
        self.norm = nn.BatchNorm1d(input_dim)
        layers = [nn.Linear(input_dim, hidden_sizes[0]), nn.Tanh()]
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_sizes[-1], output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return torch.sigmoid(self.model(x))

    def train_model(self, dataset: torch.Tensor, labels: torch.Tensor,
                    epochs: int = 10, lr: float = 1e-3, batch_size: int = 32,
                    optimizer_cls=optim.Adam):
        dataloader = DataLoader(TensorDataset(dataset, labels), batch_size=batch_size, shuffle=True)
        optimizer = optimizer_cls(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} loss: {epoch_loss/len(dataloader):.4f}")

__all__ = ["QCNNHybrid"]
