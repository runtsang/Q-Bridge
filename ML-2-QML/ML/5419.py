import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------- #
# 1. Classical RBF kernel layer with trainable support vectors
# --------------------------------------------------------------------------- #
class KernelLayer(nn.Module):
    """Classical RBF kernel layer that maps input features to similarity
    scores against a set of learnable support vectors."""
    def __init__(self, input_dim: int, n_support: int = 16, gamma: float = 1.0):
        super().__init__()
        self.support = nn.Parameter(torch.randn(n_support, input_dim))
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim)
        diff = x.unsqueeze(1) - self.support.unsqueeze(0)  # (batch, n_support, dim)
        dist_sq = torch.sum(diff * diff, dim=-1)           # (batch, n_support)
        return torch.exp(-self.gamma * dist_sq)            # RBF kernel features

# --------------------------------------------------------------------------- #
# 2. Synthetic regression dataset (superposition‑style)
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate samples x ∈ [−1,1]^d and targets y = sin(∑x) + 0.1 cos(2∑x)."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset returning a single‑channel image and a scalar target."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        # Reshape to 1×28×28 image for compatibility with the CNN backbone
        img = self.features[index].reshape(1, 28, 28)
        return {"states": torch.tensor(img, dtype=torch.float32),
                "target": torch.tensor(self.labels[index], dtype=torch.float32)}

# --------------------------------------------------------------------------- #
# 3. Classical hybrid model: CNN → RBF kernel → regression head
# --------------------------------------------------------------------------- #
class QuantumNATHybrid(nn.Module):
    """
    Classical version of the hybrid architecture.
    Image → CNN → flatten → RBF kernel → linear head → scalar output.
    """
    def __init__(self,
                 num_features: int = 16,
                 n_support: int = 16,
                 gamma: float = 1.0):
        super().__init__()
        # Convolutional feature extractor (borrowed from the original QuantumNAT)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # After two 2×2 poolings on 28×28, we obtain 16×7×7 feature maps
        self.kernel_layer = KernelLayer(16 * 7 * 7, n_support, gamma)
        self.head = nn.Linear(n_support, 1)
        self.norm = nn.BatchNorm1d(n_support)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)                     # → (bsz, 16, 7, 7)
        flat = feats.view(bsz, -1)                   # → (bsz, 784)
        kfeat = self.kernel_layer(flat)              # → (bsz, n_support)
        kfeat = self.norm(kfeat)
        out = self.head(kfeat)                       # → (bsz, 1)
        return out.squeeze(-1)                       # → (bsz,)

__all__ = ["QuantumNATHybrid", "RegressionDataset", "generate_superposition_data"]
