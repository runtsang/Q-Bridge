import torch
from torch import nn
import torch.nn.functional as F

class HybridConvSampler(nn.Module):
    """
    A hybrid classical module that merges a convolutional filter with a sampler network.
    The convolution operates on 2‑D data, producing a mean activation that feeds into
    a small feed‑forward sampler.  This mirrors the structure of the two reference
    seeds while providing a single drop‑in replacement.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        sampler_in_dim: int = 2,
        sampler_hidden: int = 4,
        sampler_out_dim: int = 2,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        self.sampler = nn.Sequential(
            nn.Linear(sampler_in_dim, sampler_hidden),
            nn.Tanh(),
            nn.Linear(sampler_hidden, sampler_out_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (..., H, W) with values in [0, 1].

        Returns:
            conv_act: Convolutional activations after sigmoid thresholding.
            sampler_out: Softmaxed sampler output based on the mean activation.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)

        conv = self.conv(x)
        conv_act = torch.sigmoid(conv - self.conv_threshold)
        mean_act = conv_act.mean(dim=[2, 3])  # (B, 1)
        sampler_out = F.softmax(self.sampler(mean_act), dim=-1)
        return conv_act, sampler_out

__all__ = ["HybridConvSampler"]
