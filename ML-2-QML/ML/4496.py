import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Iterable, Callable

class HybridQuanvolutionClassifier(nn.Module):
    """Classical counterpart of the hybrid quanvolution model.

    Features:
    * 2×2 patch extraction via a depthwise convolution.
    * RBF kernel utility for similarity analysis.
    * FastEstimator support for batch evaluation with optional Gaussian noise.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, gamma: float = 1.0):
        super().__init__()
        self.patch_conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2, bias=False)
        self.feature_dim = 4 * 14 * 14
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.gamma = gamma

    def forward(self, x: Tensor) -> Tensor:
        # Extract 2×2 patches as 4‑channel feature maps
        patches = self.patch_conv(x)
        # Flatten per sample
        return patches.view(x.size(0), -1)

    def classify(self, x: Tensor) -> Tensor:
        features = self.forward(x)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

    def kernel_matrix(self, a: Tensor, b: Tensor) -> torch.Tensor:
        """Return RBF kernel matrix between two batches of feature vectors."""
        a = self.forward(a)
        b = self.forward(b)
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        dist_sq = (diff * diff).sum(dim=-1)
        return torch.exp(-self.gamma * dist_sq)

    def evaluate(
        self,
        observables: Iterable[Callable[[Tensor], Tensor]],
        parameter_sets: List[List[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate observables on batches of parameters with optional Gaussian noise."""
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.classify(inputs)
                row = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        if shots is None:
            return results
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        noisy = []
        for row in results:
            noisy_row = [float(torch.normal(mean, max(1e-6, 1 / shots), generator=rng)) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridQuanvolutionClassifier"]
