import torch
from torch import nn

class ConvolutionalFilter(nn.Module):
    """
    A versatile 2‑D convolutional filter supporting:
    * Kernel sizes 2 or 3 (default 2).
    * Optional depth‑wise separable conv for 3×3 kernels.
    * A learnable gating threshold.
    * Batched processing.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 depthwise: bool = False,
                 use_gate: bool = True,
                 init_threshold: float = 0.0) -> None:
        super().__init__()
        assert kernel_size in (2, 3), "kernel_size must be 2 or 3"

        self.kernel_size = kernel_size
        self.depthwise = depthwise
        self.use_gate = use_gate

        if kernel_size == 2:
            self.conv = nn.Conv2d(1, 1, kernel_size=2, bias=True)
        else:  # kernel_size == 3
            self.conv = nn.Conv2d(1, 1, kernel_size=3, groups=1 if depthwise else None, bias=True)

        if self.use_gate:
            self.threshold = nn.Parameter(torch.tensor(init_threshold, dtype=torch.float32))
        else:
            self.register_buffer("threshold", torch.tensor(init_threshold))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W) or (1, H, W).

        Returns
        -------
        torch.Tensor
            Mean activation after convolution, gating, and sigmoid.
        """
        if x.ndim == 3:  # add batch dimension
            x = x.unsqueeze(0)
        logits = self.conv(x)
        gated = torch.sigmoid(logits - self.threshold)
        return gated.mean(dim=[1, 2, 3]).squeeze()

    def classify(self, x: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
        """
        Simple classification head that maps the filter output to class logits.
        """
        act = self.forward(x)
        logits = act.unsqueeze(-1).expand(-1, num_classes)
        return logits
