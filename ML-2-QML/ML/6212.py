from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionSamplerNet(nn.Module):
    """
    Classical hybrid model that replaces the standard convolutional layer
    with a 2×2 patch‑based quanvolution filter and augments the linear
    head with a lightweight sampler network.  The sampler network
    serves as a probabilistic classifier and can be swapped for a
    fully‑connected head without changing the API.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # 2×2 patch extraction via a 1×1 convolution (stride 2)
        self.patch_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        # Sampler network head
        self.sampler = nn.Sequential(
            nn.Linear(4 * 14 * 14, 64),
            nn.Tanh(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract 2×2 patches
        patches = self.patch_conv(x)
        # Flatten to feed into sampler head
        flat = patches.view(patches.size(0), -1)
        logits = self.sampler(flat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionSamplerNet"]
