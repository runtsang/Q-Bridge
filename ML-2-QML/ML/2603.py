import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalQuanvolutionFilter(nn.Module):
    """Classical 2x2 convolution that reduces a 28x28 image to 14x14 feature map."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class ClassicalFCL(nn.Module):
    """Classical surrogate for a quantum fully‑connected layer."""
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        values = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

class QuanvolutionClassifier(nn.Module):
    """Classifier that chains a classical quanvolution filter with a classical FCL head."""
    def __init__(self):
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter()
        self.fcl = ClassicalFCL()
        self.head = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor, thetas: np.ndarray) -> torch.Tensor:
        features = self.filter(x)
        q_out = self.fcl.run(thetas)
        scale = 1 + q_out
        scale = torch.tensor(scale, device=x.device).unsqueeze(0).repeat(features.size(0), 1)
        features = features * scale
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionClassifier"]
