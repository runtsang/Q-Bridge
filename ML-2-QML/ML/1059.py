import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class QuanvolutionFilter(nn.Module):
    """Hybrid classical‑quantum filter with learnable quantum‑inspired kernel and residual skip."""
    def __init__(self, n_wires: int = 4, n_layers: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(n_wires, n_wires)) for _ in range(n_layers)])
        self.bias = nn.Parameter(torch.randn(n_wires))
    def _quantum_forward(self, patch: torch.Tensor) -> torch.Tensor:
        # patch: (bsz, 4)
        x = patch
        for w in self.weights:
            x = torch.matmul(x, w)
            x = torch.tanh(x)
        x = x + self.bias
        return x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r+2, c:c+2].view(bsz, 4)
                qfeat = self._quantum_forward(patch)
                feat = qfeat + patch
                patches.append(feat)
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """Classifier that uses the QuanvolutionFilter followed by a linear head."""
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, n_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
