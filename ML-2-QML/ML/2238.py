import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Tuple

class QuantumClassifierModel(nn.Module):
    """
    Classical hybrid classifier that optionally uses a quantum kernel.
    The network architecture mirrors the quantum ansatz: a stack of linear
    layers followed by a two‑class linear head.  The `use_quantum_kernel`
    flag determines whether the input is passed through a classical RBF
    kernel or a quantum kernel computed by a separate QML module.
    """
    def __init__(self,
                 num_features: int,
                 depth: int = 2,
                 use_quantum_kernel: bool = False,
                 gamma: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.use_quantum_kernel = use_quantum_kernel
        self.gamma = gamma

        # Metadata that mirrors the quantum interface
        self.encoding = list(range(num_features))
        self.weight_sizes = []

        layers: list[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            self.weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features

        self.head = nn.Linear(in_dim, 2)
        layers.append(self.head)
        self.weight_sizes.append(self.head.weight.numel() + self.head.bias.numel())

        self.network = nn.Sequential(*layers)
        self.observables = list(range(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass through the feed‑forward network."""
        return self.network(x)

    # ------------------------------------------------------------------
    # Kernel utilities – these are fully classical but expose the same
    # interface as the quantum kernel implementation.
    # ------------------------------------------------------------------
    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value between two samples.
        If `use_quantum_kernel` is True, this method raises
        NotImplementedError because the quantum computation must be
        performed by the QML counterpart.
        """
        if self.use_quantum_kernel:
            raise NotImplementedError(
                "Quantum kernel evaluation must be performed by the QML module."
            )
        return self._rbf_kernel(x, y)

    def kernel_matrix(self,
                      a: Iterable[torch.Tensor],
                      b: Iterable[torch.Tensor]) -> np.ndarray:
        """
        Construct the Gram matrix between two collections of samples.
        """
        return np.array([[self.compute_kernel(x, y).item() for y in b] for x in a])

    # ------------------------------------------------------------------
    # Compatibility helpers – mirror the original seed API
    # ------------------------------------------------------------------
    def build_classifier_circuit(self) -> Tuple[nn.Module,
                                               Iterable[int],
                                               Iterable[int],
                                               list[int]]:
        """
        Return the network, encoding indices, weight sizes and observables
        so that callers can introspect the model structure.
        """
        return self.network, self.encoding, self.weight_sizes, self.observables

__all__ = ["QuantumClassifierModel"]
