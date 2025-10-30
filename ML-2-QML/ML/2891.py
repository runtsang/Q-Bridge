"""Hybrid classical-quantum Nat model combining CNN backbone and quantum kernel."""
import torch
import torch.nn as nn

# Import the quantum kernel from the quantum module.  
# The quantum module is expected to be in the same package namespace.
from.QuantumNAT__gen155_qml import QuantumKernel


class HybridNATModel(nn.Module):
    """Hybrid model: classical CNN feature extractor + quantum kernel + linear classifier."""

    def __init__(self, num_classes: int = 4, num_support: int = 100, feature_dim: int = 4):
        super().__init__()
        # CNN backbone: simple 2‑layer conv net followed by a 4‑dimensional projection.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, feature_dim),
            nn.ReLU(),
        )
        # Quantum kernel module (classical code imports quantum routine).
        self.q_kernel = QuantumKernel()
        # Learnable support vectors for kernel evaluation.
        self.register_parameter(
            "support_vectors", nn.Parameter(torch.randn(num_support, feature_dim))
        )
        # Linear classifier operating in kernel space.
        self.classifier = nn.Linear(num_support, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract 4‑dimensional features from the CNN.
        features = self.cnn(x)  # shape: (batch, 4)
        batch_size = features.shape[0]
        # Compute quantum kernel matrix between batch and support vectors.
        kernel_vals = torch.zeros(batch_size, self.support_vectors.shape[0], device=features.device)
        for i in range(batch_size):
            for j in range(self.support_vectors.shape[0]):
                kernel_vals[i, j] = self.q_kernel(features[i], self.support_vectors[j])
        # Pass through linear classifier to obtain logits.
        logits = self.classifier(kernel_vals)
        return logits


__all__ = ["HybridNATModel"]
