"""Combined classical self‑attention with convolutional feature extractor.

This class fuses the classical self‑attention pattern from the original
SelfAttention seed with the CNN + FC backbone of Quantum‑NAT.  It accepts
parameter arrays for the attention mechanism and returns the attended
features.  The architecture is fully differentiable and can be trained
with standard PyTorch optimisers.

Design points:
* The input is first passed through a 2‑D convolutional encoder (mirroring
  the `QFCModel.features` block).  The encoder outputs a flattened
  representation that is fed into the attention module.
* The attention module uses the supplied rotation and entangle parameters
  to compute query/key/value tensors, exactly as in the classical seed.
* An additional linear projection (like the `QFCModel.fc` block) maps the
  attended output to a fixed dimensionality (default 4) which is then
  normalised by batch‑norm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionGen172(nn.Module):
    def __init__(self, embed_dim:int=4, conv_channels:int=8, out_features:int=4):
        super().__init__()
        # Convolutional encoder (same as QFCModel.features)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels, conv_channels*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Flatten size depends on input spatial size; assume 28x28 images
        self.flat_dim = conv_channels*2 * 7 * 7
        # Linear projection to embed_dim
        self.proj = nn.Linear(self.flat_dim, embed_dim)
        # Attention parameters will be supplied externally; we keep placeholders
        self.embed_dim = embed_dim
        self.out_features = out_features
        # Final FC to produce desired output size
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_features)
        )
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x:torch.Tensor, rotation_params:np.ndarray,
                entangle_params:np.ndarray) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, 1, H, W)
            rotation_params: array of shape (embed_dim*3,) used to form query
            entangle_params: array of shape (embed_dim-1,) used to form key
        Returns:
            Tensor of shape (B, out_features)
        """
        bsz = x.size(0)
        feat = self.encoder(x)                # (B, C, H', W')
        flat = feat.view(bsz, -1)             # (B, flat_dim)
        embed = self.proj(flat)               # (B, embed_dim)

        # Prepare rotation and entangle matrices
        rot_mat = torch.tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32, device=x.device)
        ent_mat = torch.tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32, device=x.device)

        query = torch.matmul(embed, rot_mat)  # (B, embed_dim)
        key   = torch.matmul(embed, ent_mat)  # (B, embed_dim)
        value = embed                              # (B, embed_dim)

        scores = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        attended = torch.matmul(scores, value)      # (B, embed_dim)

        out = self.fc(attended)                     # (B, out_features)
        return self.norm(out)

__all__ = ["SelfAttentionGen172"]
