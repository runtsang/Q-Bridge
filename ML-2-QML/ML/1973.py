import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class FCLGen(nn.Module):
    """
    A deeper, dropout‑regularised fully‑connected layer that accepts a flat
    parameter vector ``thetas``.  The vector is parsed into the weight and
    bias tensors of a two‑layer perceptron:
        Linear -> ReLU -> Dropout -> Linear -> tanh
    The ``run`` method returns the mean activation over a batch of inputs
    constructed from ``thetas``.  This interface mirrors the original seed
    but supports batched inference and stochastic regularisation.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_dim: int = 16,
        dropout: float = 0.1,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device) if device else torch.device("cpu")
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # initialise weights to match the seed expectation (~tanh output)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # thetas are expected to be a flat vector
        # reshape to (1, n_features)
        x = thetas.view(1, -1).to(self.device)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc2(x))
        return x.mean(dim=0)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Accepts an iterable of parameters, converts to a torch tensor,
        runs forward, detaches and returns a NumPy scalar.
        """
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.forward(theta_tensor)
        return out.cpu().numpy()

__all__ = ["FCLGen"]
