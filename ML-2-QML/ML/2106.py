import torch
from torch import nn
import numpy as np

class FullyConnectedLayer(nn.Module):
    """Fully connected neural layer with tanh activation and trainable parameters."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """Compute output expectation from input thetas."""
        return self.net(thetas)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Convenience wrapper for NumPy input."""
        with torch.no_grad():
            tensor = torch.as_tensor(thetas, dtype=torch.float32)
            out = self.forward(tensor)
            return out.detach().cpu().numpy()

    def get_params(self) -> dict:
        """Return current parameters as a dictionary."""
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}

    def get_grads(self) -> dict:
        """Return gradients of parameters after a backward pass."""
        return {k: v.grad.detach().cpu().numpy() if v.grad is not None else None
                for k, v in self.named_parameters()}

__all__ = ["FullyConnectedLayer"]
