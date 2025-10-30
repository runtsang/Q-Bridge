import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Try to import the quantum components; fall back to simple placeholders
try:
    from.quanvolution_qml import QuanvolutionFilterQuantum, HybridQuantumHead
except Exception:
    class QuanvolutionFilterQuantum(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

        def forward(self, x):
            return self.conv(x).view(x.size(0), -1)

    class HybridQuantumHead(nn.Module):
        def __init__(self, in_features, shift=0.0, n_qubits=1, shots=100):
            super().__init__()
            self.linear = nn.Linear(in_features, 1)
            self.shift = shift

        def forward(self, x):
            logits = self.linear(x).squeeze()
            return torch.sigmoid(logits + self.shift)

class QuanvolutionHybridClassifier(nn.Module):
    """
    A hybrid classifier that can operate in three modes:

    * ``classic`` – 2‑D classical convolution followed by a linear head.
    * ``quantum`` – quantum patch extractor followed by a linear head.
    * ``hybrid``  – quantum patch extractor followed by a hybrid quantum‑classical head.
    """

    def __init__(self,
                 mode: str = "hybrid",
                 n_qubits: int = 4,
                 backend=None,
                 shots: int = 100,
                 shift: float = np.pi / 2,
                 num_classes: int = 10):
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes

        if mode == "classic":
            self.filter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
            self.head = nn.Linear(4 * 14 * 14, num_classes)

        elif mode == "quantum":
            self.filter = QuanvolutionFilterQuantum()
            self.head = nn.Linear(4 * 14 * 14, num_classes)

        elif mode == "hybrid":
            self.filter = QuanvolutionFilterQuantum()
            # For binary classification the hybrid head outputs a single probability.
            self.head = HybridQuantumHead(4 * 14 * 14,
                                          shift=shift,
                                          n_qubits=n_qubits,
                                          shots=shots)

        else:
            raise ValueError(f"Unsupported mode {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. ``x`` is expected to be a batch of grayscale images
        of shape (batch, 1, 28, 28).
        """
        if self.mode == "classic":
            features = self.filter(x)
            features = features.view(features.size(0), -1)
            logits = self.head(features)
            return F.log_softmax(logits, dim=-1)

        # quantum or hybrid modes
        features = self.filter(x)  # already flattened by the quantum filter
        if isinstance(self.head, nn.Linear):
            logits = self.head(features)
            return F.log_softmax(logits, dim=-1)

        # hybrid head returns a probability for one class
        prob = self.head(features)
        return torch.cat((prob, 1 - prob), dim=-1)

__all__ = ["QuanvolutionHybridClassifier"]
