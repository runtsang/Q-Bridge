import torch
from torch import nn
import numpy as np

class QCNNKernelHybrid(nn.Module):
    """
    Hybrid classical‑quantum model that combines a convolution‑like feature extractor
    with an RBF kernel for similarity evaluation. The feature extractor mirrors
    the structure of the QCNN quantum circuit, enabling a direct comparison
    between classical and quantum representations.
    """
    def __init__(self,
                 input_dim: int = 8,
                 rbf_gamma: float = 1.0) -> None:
        super().__init__()
        # Feature extractor (classical analogue of quantum convolution layers)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 4), nn.Tanh()
        )
        # Classification head
        self.head = nn.Linear(4, 1)

        # RBF kernel parameters
        self.gamma = rbf_gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor and classification head.
        """
        features = self.feature_extractor(x)
        output = torch.sigmoid(self.head(features))
        return output

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the radial‑basis‑function kernel between two feature vectors.
        """
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """
        Compute the Gram matrix between two batches of inputs using the RBF kernel.
        """
        return np.array([[self.rbf_kernel(a_i, b_j).item() for b_j in b] for a_i in a])

__all__ = ["QCNNKernelHybrid"]
