import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedQuanvolution(nn.Module):
    """Depthwise separable quanvolution filter with attention and linear head."""

    def __init__(self):
        super().__init__()
        # Learnable patch scaling factor (placeholder for dynamic patch size)
        self.patch_scale = nn.Parameter(torch.tensor(1.0))
        # Depthwise conv: per channel conv
        self.depthwise_conv = nn.Conv2d(1, 1, kernel_size=2, stride=2, groups=1, bias=False)
        # Pointwise conv to increase channel dimension
        self.pointwise_conv = nn.Conv2d(1, 4, kernel_size=1, bias=False)
        # Attention MLP to weight spatial patches
        self.attention = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 4, kernel_size=1),
            nn.Sigmoid()
        )
        # Linear classifier
        self.classifier = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, 28, 28]
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        attn = self.attention(out)
        out = out * attn
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["EnhancedQuanvolution"]
