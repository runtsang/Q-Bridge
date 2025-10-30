import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionNet(nn.Module):
    """
    Depth‑wise separable quanvolution network with residual connections.
    The filter processes 2×2 patches via a depth‑wise 2×2 convolution,
    followed by a point‑wise 1×1 convolution that fuses the 4 channels.
    A skip connection adds the input (down‑sampled) to the fused output.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_width: int = 4):
        super().__init__()
        # Depth‑wise 2×2 convolution (groups=in_channels)
        self.dw_conv = nn.Conv2d(
            in_channels, base_width, kernel_size=2, stride=2, groups=in_channels, bias=False
        )
        # Point‑wise 1×1 convolution to mix channels
        self.pw_conv = nn.Conv2d(base_width, base_width, kernel_size=1, bias=False)
        # Residual 1×1 convolution to match dimensions and down‑sample
        self.res_conv = nn.Conv2d(
            in_channels, base_width, kernel_size=1, stride=2, bias=False
        )
        self.act = nn.ReLU(inplace=True)
        # Linear head
        self.classifier = nn.Linear(base_width * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depth‑wise and point‑wise processing
        dw = self.dw_conv(x)
        pw = self.pw_conv(dw)
        # Residual connection
        res = self.res_conv(x)
        out = self.act(pw + res)
        features = out.view(out.size(0), -1)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
