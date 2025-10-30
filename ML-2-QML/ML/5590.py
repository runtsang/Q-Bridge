import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Simple 2x2 convolutional filter producing 4 feature maps."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class SamplerQNN(nn.Module):
    """
    Classical sampler network that optionally uses a Quanvolution filter
    before a linear head.  Mirrors the original SamplerQNN while adding
    the ability to process 2‑D images with a lightweight convolution.
    """
    def __init__(self, input_dim=2, hidden_dim=4, num_classes=2, use_conv=False):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.qfilter = QuanvolutionFilter()
            # 28x28 input -> 14x14 feature maps -> 4*14*14 features
            self.linear = nn.Linear(4 * 14 * 14, num_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            features = self.qfilter(x)
            features = features.view(features.size(0), -1)
            logits = self.linear(features)
        else:
            logits = self.net(x)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNN"]
