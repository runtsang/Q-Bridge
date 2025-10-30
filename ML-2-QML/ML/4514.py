"""Hybrid kernel classifier combining convolution, self‑attention and RBF kernel.

The implementation mirrors the quantum version in QuantumKernelMethod__gen198_qml.py
while staying fully classical.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import seed helpers
from Conv import Conv
from SelfAttention import SelfAttention
from QuantumKernelMethod import Kernel

class HybridKernelClassifier(nn.Module):
    """Classical hybrid classifier that fuses convolution, self‑attention and RBF kernel."""
    def __init__(self,
                 conv_kernel_size: int = 2,
                 conv_threshold: float = 0.0,
                 embed_dim: int = 4,
                 rbf_gamma: float = 1.0,
                 num_classes: int = 2) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.conv = Conv()
        # Self‑attention
        self.attn = SelfAttention()
        # RBF kernel
        self.kernel = Kernel(gamma=rbf_gamma)
        # Dense head
        self.fc = nn.Linear(in_features=embed_dim + 1, out_features=num_classes)

        # Prototype vector for kernel evaluation
        self.register_buffer('prototype', torch.randn((1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Logits for each class.
        """
        batch = x.shape[0]
        # Flatten each image to a 2D array of shape (conv_kernel_size, conv_kernel_size)
        # For simplicity, we crop the top‑left corner.
        conv_inputs = x[:, :, :self.conv.kernel_size, :self.conv.kernel_size]
        conv_vals = []
        for i in range(batch):
            arr = conv_inputs[i].squeeze().cpu().numpy()
            conv_vals.append(self.conv.run(arr))
        conv_tensor = torch.tensor(conv_vals, device=x.device).unsqueeze(-1)  # shape (batch, 1)

        # Self‑attention on the convolution output
        # Expand conv_tensor to match embed_dim
        attn_inputs = conv_tensor.repeat(1, self.attn.embed_dim)
        rotation = np.eye(self.attn.embed_dim, dtype=np.float32)
        entangle = np.eye(self.attn.embed_dim, dtype=np.float32)
        attn_output = torch.tensor(
            self.attn.run(rotation, entangle, attn_inputs.cpu().numpy()),
            device=x.device
        )  # shape (batch, embed_dim)

        # RBF kernel between conv output and prototype
        proto = self.prototype.expand(batch, -1)
        kernel_vals = torch.tensor(
            [self.kernel(conv_tensor[i], proto[i]).item() for i in range(batch)],
            device=x.device
        ).unsqueeze(-1)  # shape (batch, 1)

        # Concatenate attention and kernel features
        features = torch.cat([attn_output, kernel_vals], dim=-1)  # shape (batch, embed_dim+1)

        # Dense head
        logits = self.fc(features)
        return F.softmax(logits, dim=-1)

__all__ = ["HybridKernelClassifier"]
