"""ConvGen – classical convolution filter with optional quantum fallback.

This class implements a drop‑in replacement for the original Conv filter.
It supports multi‑channel inputs, a learnable threshold, and can be
configured to use a quantum backend via the ``use_quantum`` flag.
"""

import torch
from torch import nn

__all__ = ["ConvGen"]

class ConvGen(nn.Module):
    """
    Classical convolution filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.
    threshold : float or None, default 0.0
        Threshold value applied to the convolution output before
        the sigmoid activation. If ``None`` the threshold is not applied.
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 1
        Number of output channels.
    bias : bool, default True
        Whether to add a bias term.
    use_quantum : bool, default False
        When ``True`` the class will raise ``NotImplementedError`` in
        :meth:`forward` because the quantum implementation lives in
        the QML module.
    device : str, default "cpu"
        Device on which the model is allocated.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float | None = 0.0,
        in_channels: int = 1,
        out_channels: int = 1,
        bias: bool = True,
        use_quantum: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_quantum = use_quantum
        self.device = device

        if not use_quantum:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, bias=bias
            ).to(device)
            nn.init.xavier_uniform_(self.conv.weight)
            if bias:
                nn.init.zeros_(self.conv.bias)
        else:
            self.conv = None  # placeholder for quantum path

        if threshold is not None:
            self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        else:
            self.threshold = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classical filter.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Filtered output of shape (batch, out_channels, H', W').
        """
        if self.use_quantum:
            raise NotImplementedError(
                "Quantum forward pass is only available in the QML module."
            )
        out = self.conv(x)
        if self.threshold is not None:
            out = torch.sigmoid(out - self.threshold)
        return out

    def run(self, data: torch.Tensor | list | tuple) -> float:
        """
        Run a single sample through the filter and return the average
        activation value.  ``data`` can be a NumPy array or a torch
        tensor; it is internally converted to a tensor of shape
        (1, in_channels, H, W).

        Parameters
        ----------
        data : torch.Tensor | list | tuple
            Input sample.

        Returns
        -------
        float
            Mean activation after the convolution and optional
            thresholding.
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            # Ensure shape: (1, in_channels, H, W)
            if data.ndim == 2:
                data = data.unsqueeze(0).unsqueeze(0)
            elif data.ndim == 3:
                data = data.unsqueeze(0)
            out = self.forward(data.to(self.device))
            return out.mean().item()
