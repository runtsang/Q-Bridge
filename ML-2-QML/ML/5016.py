import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridNAT(nn.Module):
    """Hybrid classical network that optionally delegates the final head to a quantum‑style head.

    The backbone mirrors the original QuantumNAT CNN, but the output head can be
    replaced with a lightweight parametric function that mimics a quantum
    expectation.  A small RBF kernel and a 2‑layer regressor are bundled as
    utilities for downstream experiments.
    """

    class RBFKernel(nn.Module):
        """Simple radial‑basis‑function kernel."""
        def __init__(self, gamma: float = 1.0):
            super().__init__()
            self.gamma = gamma

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def __init__(self,
                 mode: str = "classification",
                 use_quantum_head: bool = False,
                 shift: float = 0.0,
                 kernel_gamma: float = 1.0):
        super().__init__()
        self.mode = mode
        self.use_quantum_head = use_quantum_head
        self.shift = shift
        self.kernel_gamma = kernel_gamma

        # Backbone – identical to the classical part of QFCModel
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flattened size 16 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

        # Classical head
        self.class_head = nn.Linear(4, 1)

        # Quantum‑style head – a simple sigmoid with parameterised shift
        self.quantum_head = nn.Linear(4, 1)

        # Simple RBF kernel
        self.rbf = self.RBFKernel(kernel_gamma)

        # Small regression sub‑network (EstimatorQNN)
        self.regressor = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional quantum‑style head."""
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        out = self.fc(flat)
        out = self.norm(out)

        if self.mode == "regression":
            # For regression we use the regressor on the first two features
            reg_in = flat[:, :2]
            return self.regressor(reg_in)

        if self.use_quantum_head:
            # Mimic a quantum expectation with a shifted sigmoid
            logits = self.quantum_head(out)
            return torch.sigmoid(logits + self.shift)

        # Default classification head
        logits = self.class_head(out)
        return torch.sigmoid(logits + self.shift)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Compute pairwise RBF kernel matrix between two batches."""
        return np.array([[self.rbf(x, y).item() for y in b] for x in a])

    def predict_pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return quantum‑style similarity between two single samples."""
        # For demo use the first feature of each sample
        x_feat = x.view(-1, 1)[:1]
        y_feat = y.view(-1, 1)[:1]
        return self.rbf(x_feat, y_feat)

__all__ = ["HybridNAT"]
