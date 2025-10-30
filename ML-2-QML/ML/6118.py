import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """
    Classical quanvolution hybrid that uses a depthwise 2×2 convolution
    followed by a small MLP head. All parameters are trainable with
    standard back‑prop.
    """
    def __init__(self, in_channels=1, out_channels=4, patch_size=2, stride=2, mlp_hidden=128):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride)
        # Compute number of patches per image: (28 // patch_size)^2
        n_patches = (28 // patch_size) ** 2
        # Flattened feature dimension: out_channels * n_patches
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * n_patches, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 10)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        logits = self.mlp(features)
        return F.log_softmax(logits, dim=-1)
