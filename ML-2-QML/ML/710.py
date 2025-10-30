import torch
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin

class EstimatorQNN(BaseEstimator, RegressorMixin, nn.Module):
    """
    A configurable feed‑forward regression network that implements the
    scikit‑learn estimator API.  It supports arbitrary hidden layer sizes
    and activation functions, and includes a simple training loop.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_sizes: tuple[int,...] = (16, 8, 4),
                 activation=nn.Tanh):
        super().__init__()
        self.input_dim = input_dim
        layers = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(activation())
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(self, X, y,
            epochs: int = 200,
            lr: float = 1e-3,
            batch_size: int = 32,
            verbose: bool = False):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.forward(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
            if verbose and (epoch + 1) % 20 == 0:
                print(f"[epoch {epoch+1}] loss={loss.item():.4f}")
        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self.forward(torch.tensor(X, dtype=torch.float32)).numpy().flatten()

__all__ = ["EstimatorQNN"]
