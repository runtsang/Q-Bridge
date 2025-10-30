import torch
from torch import nn
import torch.nn.functional as F

class Conv(nn.Module):
    """
    Classical hybrid convolution filter.
    Implements a depth‑wise separable convolution followed by a
    point‑wise 1×1 convolution. The activation threshold is a learnable
    parameter that can be frozen if desired. The forward method
    accepts a batch of grayscale images and returns a probability‑like
    score in the range [0, 1].
    """

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 bias: bool = True,
                 padding: str ='same'):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        padding_mode = 0 if padding == 'valid' else kernel_size // 2
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size,
                                   groups=1, bias=bias, padding=padding_mode)
        self.pointwise = nn.Conv2d(1, 1, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images (N, 1, H, W).

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) with values in [0, 1].
        """
        out = self.depthwise(x)
        out = torch.sigmoid(out - self.threshold)
        out = self.pointwise(out)
        # Global average pooling
        return out.view(out.size(0), -1).mean(dim=1)

    def set_threshold(self, value: float) -> None:
        """Set the learnable threshold."""
        self.threshold.data.fill_(value)

    def freeze_threshold(self) -> None:
        """Freeze the threshold parameter."""
        self.threshold.requires_grad = False
