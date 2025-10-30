import numpy as np
import torch
from torch import nn

class HybridFCL(nn.Module):
    """
    Hybrid fully connected layer combining a classical linear transformation
    with optional quantum kernel evaluation.  The class is compatible with
    the original FCL interface and extends it with:
        * RBF kernel support via the ``kernel`` argument.
        * Optional quantum kernel fallback.
        * Regression or classification heads toggled by ``mode``.
    """
    def __init__(self,
                 n_features: int = 1,
                 mode: str = "regression",
                 kernel: str = "rbf",
                 n_qubits: int = 4,
                 depth: int = 2,
                 backend: str = "qasm_simulator",
                 shots: int = 512):
        super().__init__()
        self.n_features = n_features
        self.mode = mode
        self.kernel_type = kernel
        self.linear = nn.Linear(n_features, 1)
        if mode == "classification":
            self.head = nn.Linear(1, 2)
        else:
            self.head = nn.Identity()
        # Parameters for quantum part (only stored for API compatibility)
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Mimic the original FCL.run: apply a linear layer, tanh, and return a scalar
        expectation value.  ``thetas`` can be any one‑dimensional array.
        """
        x = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
        z = torch.tanh(self.linear(x)).mean(dim=0)
        return z.detach().cpu().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that optionally applies a kernel transformation before
        the linear head.  For ``kernel='rbf'`` a classical RBF kernel is used;
        otherwise the raw input is forwarded unchanged.
        """
        if self.kernel_type == "rbf":
            # Compute the mean RBF kernel between the input and itself
            diff = x.unsqueeze(1) - x.unsqueeze(0)
            gamma = 1.0
            k = torch.exp(-gamma * torch.sum(diff * diff, dim=-1))
            features = k.mean(dim=1, keepdim=True)
        else:
            features = x
        out = self.linear(features)
        out = self.head(out)
        return out.squeeze(-1)

    @staticmethod
    def kernel_matrix(a: torch.Tensor, b: torch.Tensor, gamma: float = 1.0) -> np.ndarray:
        """
        Classical RBF kernel matrix between two batches.
        """
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        k = torch.exp(-gamma * torch.sum(diff * diff, dim=-1))
        return k.detach().cpu().numpy()

    @classmethod
    def generate_data(cls, num_features: int, samples: int):
        """
        Generate synthetic regression data (mirroring the quantum example).
        """
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

__all__ = ["HybridFCL"]
