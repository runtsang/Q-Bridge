import torch
from torch import nn
import numpy as np

class HybridFCL(nn.Module):
    """A classical multi‑layer feed‑forward network that mimics a fully‑connected quantum layer.

    The architecture is inspired by the EstimatorQNN example and the simple
    fully‑connected layer in the original FCL.  The network accepts an arbitrary
    number of input features, passes them through two hidden layers with
    Tanh activations, and produces a single scalar output.  The class exposes
    a ``run`` method that mirrors the API of the quantum counterpart, making
    it trivial to swap the implementation in a hybrid experiment.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int, int] = (8, 4), output_dim: int = 1, activation=nn.Tanh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            activation(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            activation(),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

    def run(self, inputs: np.ndarray | torch.Tensor) -> np.ndarray:
        """Convenience wrapper that accepts a NumPy array or a torch Tensor.

        The method returns a NumPy array containing the network output.
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs.astype(np.float32))
        return self.forward(inputs).detach().numpy()

__all__ = ["HybridFCL"]
