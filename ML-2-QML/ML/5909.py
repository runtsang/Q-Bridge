"""Hybrid classical neural network combining a quanvolution filter and a sampler network."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionSamplerHybrid(nn.Module):
    """
    A hybrid model that fuses a classical quanvolution-like convolutional filter
    with a lightweight sampler network. The convolution extracts 4 features
    per 2×2 patch, while the sampler network produces 2 probability‑like
    outputs per patch. The concatenated feature map is flattened and fed
    into a linear classifier.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Convolutional filter (4 output channels, 2×2 kernel, stride 2)
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Sampler network: 2 inputs → 2 outputs (softmax)
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        # Linear head
        self.fc = nn.Linear(4 * 14 * 14 + 2 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        batch_size = x.size(0)
        # Convolutional features
        conv_feat = self.conv(x)  # (batch, 4, 14, 14)
        conv_flat = conv_feat.view(batch_size, -1)  # (batch, 4*14*14)

        # Sampler features
        # Extract 2×2 patches
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (batch, 1, 14, 14, 2, 2)
        patches = patches.squeeze(1)  # (batch, 14, 14, 2, 2)
        tl = patches[..., 0, 0]  # top‑left
        br = patches[..., 1, 1]  # bottom‑right
        sampler_input = torch.stack([tl.squeeze(1), br.squeeze(1)], dim=-1)  # (batch, 14, 14, 2)
        sampler_input = sampler_input.reshape(-1, 2)  # (batch*14*14, 2)
        sampler_out = self.sampler(sampler_input)  # (batch*14*14, 2)
        sampler_flat = sampler_out.view(batch_size, -1)  # (batch, 2*14*14)

        # Concatenate features
        features = torch.cat([conv_flat, sampler_flat], dim=1)
        logits = self.fc(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionSamplerHybrid"]
