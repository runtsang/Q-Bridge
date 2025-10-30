import numpy as np
import torch
from torch import nn

class FullyConnectedLayer(nn.Module):
    """
    Classical fully connected layer that mimics a quantum circuit output.
    Provides a ``run`` method for inference and supports autograd for training.
    """
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear layer followed by tanh activation.
        """
        return torch.tanh(self.linear(x))

    def run(self, thetas: np.ndarray | Iterable[float]) -> np.ndarray:
        """
        Run the layer on a list/array of parameter values.
        Returns the mean activation as a NumPy array.
        """
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = self.forward(theta_tensor)
        expectation = out.mean(dim=0)
        return expectation.detach().numpy()
