import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FCL(nn.Module):
    """
    Classical fully connected layer with optional training on target data.
    """

    def __init__(self, n_features: int = 1, hidden_dim: int = 16):
        super().__init__()
        self.linear = nn.Linear(n_features, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.linear(x))
        return self.output(hidden)

    def run(self, thetas):
        """
        Compatibility wrapper: accepts an iterable of scalars and returns a numpy array of predictions.
        """
        x = torch.tensor(thetas, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            pred = self.forward(x).squeeze(1)
        return pred.numpy()

    def optimize(self, data_loader, epochs: int = 100, lr: float = 1e-3):
        """
        Train the model on a given data_loader using MSE loss.
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            for batch_x, batch_y in data_loader:
                optimizer.zero_grad()
                pred = self(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
