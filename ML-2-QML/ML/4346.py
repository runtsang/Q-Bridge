import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SamplerQNNGen020(nn.Module):
    """
    A hybrid classical sampler that integrates convolution, QCNN‑style layers,
    self‑attention, and a final softmax sampler. The architecture mirrors the
    quantum counterpart while remaining fully classical.
    """
    def __init__(self) -> None:
        super().__init__()
        # 2x2 convolution filter
        self.conv = nn.Conv2d(1, 1, kernel_size=2, bias=True)

        # Expand to 8‑dimensional feature vector (4 from flatten, 1 from conv)
        self.input_linear = nn.Linear(5, 8)

        # QCNN‑style fully connected stack
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

        # Self‑attention block (single‑head)
        self.attn = nn.MultiheadAttention(embed_dim=8, num_heads=1, batch_first=True)

        # Sampler network mapping to 2 logits
        self.sampler_net = nn.Sequential(
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 2, 2).

        Returns
        -------
        torch.Tensor
            Softmax distribution over 2 classes.
        """
        # Flatten 2x2 to 4‑dim vector
        x_flat = x.view(x.size(0), -1)          # (batch, 4)
        # Convolution output
        x_conv = self.conv(x).view(x.size(0), -1)  # (batch, 1)
        # Concatenate for 5‑dim input
        x = torch.cat([x_flat, x_conv], dim=1)   # (batch, 5)
        # Linear expansion to 8 dims
        x = self.input_linear(x)                # (batch, 8)

        # QCNN flow
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Self‑attention (sequence length 1)
        x = x.unsqueeze(1)              # (batch, 1, 8)
        attn_out, _ = self.attn(x, x, x)
        x = attn_out.squeeze(1)         # (batch, 8)

        # Sampler logits
        logits = self.sampler_net(x)    # (batch, 2)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNNGen020"]
