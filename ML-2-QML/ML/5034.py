import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Sequence, Callable

# --------------------------------------------------------------------
# Classical kernel utilities (inspired by QuantumKernelMethod)
# --------------------------------------------------------------------
class KernalAnsatz(nn.Module):
    """Radial basis function kernel. Keeps API compatible with the quantum kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wraps :class:`KernalAnsatz` to expose a kernel matrix interface."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------
# Patch encoder & estimator (combining Quanvolution + EstimatorQNN)
# --------------------------------------------------------------------
class ClassicalPatchEncoder(nn.Module):
    """Extract 2×2 patches and flatten them into a feature vector."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv: (B, 4, 14, 14) for MNIST 28x28
        patches = self.conv(x)                       # shape (B, 4, 14, 14)
        return patches.view(patches.size(0), -1)      # (B, 4*14*14)

class EstimatorNN(nn.Module):
    """Small feed‑forward regressor mirroring the EstimatorQNN example."""
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (8, 4), output_dim: int = 10):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            if b!= output_dim:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

class HybridQuanvolution(nn.Module):
    """
    Classical hybrid network that first extracts 2×2 patches with a
    convolutional layer and then feeds the flattened features into a
    small neural‑network head.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.encoder = ClassicalPatchEncoder(in_channels)
        self.head = EstimatorNN(self.encoder.conv.out_channels * 14 * 14, output_dim=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)

# --------------------------------------------------------------------
# Evaluation utilities (FastBaseEstimator + FastEstimator)
# --------------------------------------------------------------------
class FastBaseEstimator:
    """Batch‑wise model evaluation with optional shot‑noise simulation."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self, inputs: Sequence[torch.Tensor]) -> List[List[float]]:
        self.model.eval()
        with torch.no_grad():
            return [[float(self.model(x.unsqueeze(0)).mean().item())] for x in inputs]

class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to the deterministic estimates."""
    def evaluate(self, inputs: Sequence[torch.Tensor], shots: int | None = None, seed: int | None = None) -> List[List[float]]:
        raw = super().evaluate(inputs)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        return [[rng.normal(mean, max(1e-6, 1 / shots))] for mean in [row[0] for row in raw]]

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "ClassicalPatchEncoder",
    "EstimatorNN",
    "HybridQuanvolution",
    "FastEstimator",
]
