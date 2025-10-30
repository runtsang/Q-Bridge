import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class HybridQuantumLayer(nn.Module):
    """
    Classical hybrid layer that mimics a fully connected quantum layer
    combined with a simple 2×2 quanvolution filter.  The class exposes
    a ``run`` method that accepts a sequence of parameters (treated as
    the input feature vector), applies a 2×2 convolution to reshape the
    data into 14×14 patches, then feeds the flattened feature map
    through a learnable linear head and a tanh non‑linearity.
    """
    def __init__(self, n_features: int = 28*28, n_out: int = 1) -> None:
        super().__init__()
        # 2×2 filter with stride 2 (equivalent to the original quanvolution)
        self.conv = nn.Conv2d(1, 1, kernel_size=2, stride=2, bias=False)
        self.linear = nn.Linear(14*14, n_out)
        self.n_features = n_features

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Parameters
        ----------
        thetas : Iterable[float]
            Input feature vector, length must equal ``self.n_features``.
        Returns
        -------
        np.ndarray
            The scalar output of the layer after applying tanh.
        """
        if len(thetas)!= self.n_features:
            raise ValueError(f"Expected {self.n_features} elements, got {len(thetas)}")
        x = torch.as_tensor(thetas, dtype=torch.float32).view(1, 1, 28, 28)
        feat = self.conv(x)
        flat = feat.view(1, -1)
        out = self.linear(flat)
        return torch.tanh(out).detach().numpy()
