import torch
from torch import nn

class ConvGen404(nn.Module):
    """Hybrid‑compatible convolution filter with classical implementation.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.
    threshold : float, default 0.0
        Threshold applied before sigmoid activation.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply convolution and sigmoid activation.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape (H, W) or (1, H, W).

        Returns
        -------
        torch.Tensor
            Scalar in [0, 1] representing the filter output.
        """
        if data.dim() == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

__all__ = ["ConvGen404"]
