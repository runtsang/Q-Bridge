import torch
from torch import nn
import torch.nn.functional as F

class RBFLayer(nn.Module):
    """Gaussian RBF kernel layer with learnable centers."""
    def __init__(self, in_features: int, out_features: int, gamma: float = 1.0):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(out_features, in_features))
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = (diff ** 2).sum(-1)
        return torch.exp(-self.gamma * dist_sq)

class ConvFilter(nn.Module):
    """2‑D convolution filter emulating a quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

class QCNNHybridModel(nn.Module):
    """Hybrid classical network combining a convolution filter,
    an RBF layer and a simple classifier."""
    def __init__(self, image_size: int = 8, rbf_out: int = 16, gamma: float = 1.0):
        super().__init__()
        self.filter = ConvFilter(kernel_size=2, threshold=0.0)
        self.rbf = RBFLayer(in_features=image_size * image_size,
                            out_features=rbf_out,
                            gamma=gamma)
        self.classifier = nn.Sequential(
            nn.Linear(rbf_out, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, 1, H, W)
        filt_out = self.filter(x)
        flatten = filt_out.view(filt_out.size(0), -1)
        rbf_emb = self.rbf(flatten)
        logits = self.classifier(rbf_emb)
        return torch.sigmoid(logits).squeeze(-1)

def QCNNHybrid() -> QCNNHybridModel:
    """Factory returning a configured :class:`QCNNHybridModel`."""
    return QCNNHybridModel()
