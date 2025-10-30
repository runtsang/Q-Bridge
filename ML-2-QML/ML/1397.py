import torch
from torch import nn
from typing import Optional

class ConvGen361(nn.Module):
    """
    Enhanced convolutional filter that generalises the original Conv class.
    Supports multi‑channel input, optional depth‑wise separable convolution,
    learnable bias, and L2 regularisation.  The `run` method accepts a 2‑D
    numpy array or a torch tensor and returns the mean sigmoid activation.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 2,
        stride: int = 1,
        padding: int = 0,
        separable: bool = False,
        bias: bool = True,
        weight_decay: Optional[float] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.separable = separable
        if separable:
            # depth‑wise separable conv: one conv per input channel
            self.depthwise = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=bias,
                device=device,
            )
            self.pointwise = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=bias,
                device=device,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                device=device,
            )
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.separable:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)
        return x

    def run(self, data) -> float:
        """
        Apply the filter to a 2‑D array and return the mean sigmoid activation.
        The input can be a numpy array or a torch tensor.
        """
        if isinstance(data, torch.Tensor):
            tensor = data.float()
        else:
            tensor = torch.as_tensor(data, dtype=torch.float32)
        # Ensure shape (C, H, W)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        # Forward
        out = self.forward(tensor)
        activations = torch.sigmoid(out)
        return activations.mean().item()
