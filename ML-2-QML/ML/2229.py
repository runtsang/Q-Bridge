"""Hybrid classical classifier module integrating a feed‑forward network and an optional RBF kernel feature map.

The implementation mirrors the original `build_classifier_circuit` API while adding a lightweight
kernel wrapper.  The public ``HybridClassifier`` class exposes a clean interface that can be
instantiated with or without the kernel, enabling a side‑by‑side comparison with its quantum
counterpart.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Iterable, Tuple, Sequence

# --------------------------------------------------------------------------- #
#  Kernel utilities
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Classical radial basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Convenience wrapper that exposes a single ``forward`` interface."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = RBFKernel(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two collections of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
#  Classical classifier factory
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a simple fully‑connected network that mimics the signature of the quantum
    ``build_classifier_circuit`` helper.

    Returns
    -------
    network : nn.Sequential
        Feed‑forward network with ``depth`` hidden layers.
    encoding : Iterable[int]
        Index mapping that the quantum counterpart uses for data‑encoding.
    weight_sizes : Iterable[int]
        Number of trainable parameters per layer (used for bookkeeping).
    observables : list[int]
        Dummy observable list that keeps API parity with the quantum side.
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

# --------------------------------------------------------------------------- #
#  Hybrid classifier class
# --------------------------------------------------------------------------- #
class HybridClassifier(nn.Module):
    """
    Classical hybrid classifier that optionally augments the input with an RBF kernel
    similarity to a set of support vectors.

    Parameters
    ----------
    num_features : int
        Dimensionality of the raw input.
    depth : int, default 2
        Number of hidden layers in the feed‑forward network.
    use_kernel : bool, default False
        If ``True`` the model will prepend a kernel feature computed against
        ``support_vectors``.
    gamma : float, default 1.0
        RBF kernel bandwidth.
    support_vectors : torch.Tensor, optional
        Reference points used when ``use_kernel`` is ``True``.  If omitted a
        random tensor is created internally.
    """
    def __init__(
        self,
        num_features: int,
        depth: int = 2,
        use_kernel: bool = False,
        gamma: float = 1.0,
        support_vectors: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.use_kernel = use_kernel
        self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth
        )

        if use_kernel:
            self.kernel = Kernel(gamma)
            if support_vectors is None:
                support_vectors = torch.randn(5, num_features)
            self.support_vectors = support_vectors
        else:
            self.kernel = None
            self.support_vectors = None

    def _kernel_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute kernel similarity of *x* to the support set."""
        features = torch.stack([self.kernel(x, sv) for sv in self.support_vectors], dim=-1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_kernel:
            kernel_feat = self._kernel_features(x)
            x = torch.cat([x, kernel_feat], dim=-1)
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits."""
        return self.forward(x)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Convenience wrapper around the module‑level kernel_matrix helper."""
        return kernel_matrix(a, b, gamma=self.kernel.ansatz.gamma if self.kernel else 1.0)
