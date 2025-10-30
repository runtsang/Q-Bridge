"""Hybrid classical kernel‑classifier with attention and Quantum‑NAT feature extractor."""

from __future__ import annotations

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Feature extractor – Quantum‑NAT style CNN
# --------------------------------------------------------------------------- #
class QFCModel(nn.Module):
    """Convolutional backbone producing a 4‑dimensional embedding."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)

# --------------------------------------------------------------------------- #
# 2. Classical self‑attention layer
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    """Re‑implementation of the simple self‑attention block."""
    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# --------------------------------------------------------------------------- #
# 3. Radial‑basis‑function kernel (classical)
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Gaussian kernel with trainable gamma."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: Tuple[torch.Tensor,...],
                  b: Tuple[torch.Tensor,...],
                  gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix for a collection of vectors."""
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 4. Classical classifier factory (mirrors the quantum API)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int,
                             depth: int) -> Tuple[nn.Module, Tuple[int,...], Tuple[int,...], Tuple[int,...]]:
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = tuple(range(num_features))
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
    observables = tuple(range(2))
    return network, encoding, tuple(weight_sizes), observables

# --------------------------------------------------------------------------- #
# 5. Hybrid model
# --------------------------------------------------------------------------- #
class HybridKernelClassifier(nn.Module):
    """
    A hybrid classical model that:
      1) extracts 4‑D features via a CNN (Quantum‑NAT style)
      2) optionally applies a self‑attention transformation
      3) computes pairwise RBF similarities (kernel matrix)
      4) feeds the similarity vector into a shallow feed‑forward classifier
    """
    def __init__(self,
                 use_attention: bool = True,
                 attention_dim: int = 4,
                 gamma: float = 1.0,
                 depth: int = 3) -> None:
        super().__init__()
        self.feature_extractor = QFCModel()
        self.attention = ClassicalSelfAttention(attention_dim) if use_attention else None
        self.kernel = RBFKernel(gamma)
        self.classifier, _, _, _ = build_classifier_circuit(
            num_features=attention_dim if use_attention else 4,
            depth=depth
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Feature extraction
        feat = self.feature_extractor(x)           # (B, 4)

        # 2. Optional attention
        if self.attention:
            # Random parameters for demo – in practice these would be learned
            rot = np.random.randn(feat.shape[-1], feat.shape[-1])
            ent = np.random.randn(feat.shape[-1])
            feat = torch.from_numpy(
                self.attention.run(rot, ent, feat.cpu().numpy())
            ).to(feat.device)

        # 3. Kernel evaluation – use the feature itself as both arguments
        k = self.kernel(feat, feat).squeeze(-1)     # (B,)

        # 4. Classification
        logits = self.classifier(k)
        return logits

__all__ = ["HybridKernelClassifier", "build_classifier_circuit",
           "kernel_matrix", "QFCModel", "ClassicalSelfAttention", "RBFKernel"]
