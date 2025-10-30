import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """
    Classical depth‑wise separable quanvolution filter with a linear head.
    Adds feature normalisation and a hybrid loss that can be combined with a quantum loss.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2,
                 num_classes: int = 10):
        super().__init__()
        # Depth‑wise separable convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        # Feature normalisation
        self.norm = nn.LayerNorm(out_channels * 14 * 14)
        # Linear classification head
        self.linear = nn.Linear(out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, 1, 28, 28)
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.act(out)
        out = out.view(out.size(0), -1)          # Flatten
        out = self.norm(out)                     # Normalise features
        logits = self.linear(out)
        return F.log_softmax(logits, dim=-1)

    def hybrid_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                    quantum_fidelity: torch.Tensor | None = None,
                    alpha: float = 0.5) -> torch.Tensor:
        """
        Compute a hybrid loss mixing cross‑entropy with an optional quantum fidelity term.
        """
        ce = F.nll_loss(logits, targets)
        if quantum_fidelity is not None:
            return alpha * ce + (1 - alpha) * quantum_fidelity
        return ce

__all__ = ["QuanvolutionHybrid"]
