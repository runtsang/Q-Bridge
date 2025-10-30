import torch
import torch.nn as nn
import torch.nn.functional as F

# Classical convolutional network enhanced with a learnable 2x2 filter and scaling factor
class QuanvolutionEnhanced(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # First convolution: 1 channel → 8 channels, 2x2 kernel, stride 2
        self.conv1 = nn.Conv2d(1, 8, kernel_size=2, stride=2)
        # Second convolution: 8 channels → 8 channels, 1x1 kernel
        self.conv2 = nn.Conv2d(8, 8, kernel_size=1)
        # Learnable scaling factor for the feature maps
        self.scale = nn.Parameter(torch.tensor(1.0))
        # Linear classifier mapping 8×14×14 features to 10 classes
        self.linear = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)           # shape: (bsz, 8, 14, 14)
        out2 = self.conv2(out)        # shape: (bsz, 8, 14, 14)
        out = out + out2              # skip connection
        out = self.scale * out        # scaling
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)     # classification logits
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionEnhanced"]
