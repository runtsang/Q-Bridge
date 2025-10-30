import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionEnhanced(nn.Module):
    """
    Classical quanvolutional model with learnable multi‑scale convolution,
    skip‑connection, and a two‑layer MLP head.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 patch_size: int = 2, stride: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=patch_size,
                              stride=stride, bias=False)

        h_out = (28 - patch_size) // stride + 1
        self.skip_linear = nn.Linear(in_channels * 28 * 28,
                                     4 * h_out * h_out)

        self.mlp = nn.Sequential(
            nn.Linear(8 * h_out * h_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        conv_feat = self.conv(x)          # (B, 4, H', W')
        conv_flat = conv_feat.view(x.size(0), -1)

        skip_feat = self.skip_linear(x.view(x.size(0), -1))

        features = torch.cat([conv_flat, skip_feat], dim=1)
        logits = self.mlp(features)
        return F.log_softmax(logits, dim=-1)
