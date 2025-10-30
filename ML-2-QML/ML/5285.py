import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

# ---------- Classical self‑attention ----------
class SelfAttention:
    """Pure‑Python self‑attention block matching the quantum API."""
    def __init__(self, embed_dim: int = 4):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# ---------- Classical RBF kernel ----------
class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

# ---------- Classical classifier factory ----------
def build_classifier_circuit(num_features: int, depth: int):
    """Feed‑forward network that mimics the quantum classifier structure."""
    layers = []
    in_dim = num_features
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, list(range(num_features)), weight_sizes, observables

# ---------- Hybrid kernel‑classifier ----------
class QuantumHybridKernelClassifier(nn.Module):
    """
    Classical hybrid model that chains:
      1) a self‑attention preprocessing step,
      2) an RBF kernel matrix,
      3) a small feed‑forward classifier.
    """
    def __init__(self,
                 num_features: int,
                 attention_dim: int = 4,
                 kernel_gamma: float = 1.0,
                 classifier_depth: int = 2):
        super().__init__()
        self.attention = SelfAttention(embed_dim=attention_dim)
        self.kernel = Kernel(gamma=kernel_gamma)
        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_features, classifier_depth)

    def preprocess(self, X: np.ndarray, rotation_params: np.ndarray, entangle_params: np.ndarray) -> np.ndarray:
        """Apply self‑attention to raw features."""
        return self.attention.run(rotation_params, entangle_params, X)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix between two collections of feature vectors."""
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

    def forward(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        """
        If Y is None, use X as both arguments.  The kernel matrix is flattened
        and fed into the classifier.  Returns raw logits.
        """
        if Y is None:
            Y = X
        K = self.kernel_matrix(X, Y)
        out = self.classifier(torch.tensor(K, dtype=torch.float32, device=X.device))
        return out

__all__ = ["SelfAttention", "KernalAnsatz", "Kernel", "build_classifier_circuit", "QuantumHybridKernelClassifier"]
