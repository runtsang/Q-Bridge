import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvGen404(nn.Module):
    """
    Differentiable convolution filter with optional bias and L2 regularisation.
    Supports 1‑channel or 3‑channel inputs and can be trained end‑to‑end.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 bias: bool = False,
                 l2_reg: float | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l2_reg = l2_reg

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              bias=bias)

        nn.init.normal_(self.conv.weight, mean=0.0, std=1e-2)
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns a tensor of shape (batch, out_channels, H-1, W-1)
        after applying sigmoid activation.
        """
        logits = self.conv(x)
        probs = torch.sigmoid(logits - self.threshold)
        return probs

    def run(self, data) -> float:
        """
        Run the filter on a single image or a batch and return the mean activation.
        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        if data.ndim == 2:
            # Add batch and channel dimensions
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndim == 3:
            # Assume shape (H, W, C) or (C, H, W)
            if data.shape[0] in (1, 3):
                data = data.unsqueeze(0)
            else:
                data = data.permute(2, 0, 1).unsqueeze(0)
        else:
            data = data.unsqueeze(0)

        probs = self.forward(data)
        return probs.mean().item()
