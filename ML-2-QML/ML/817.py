import torch
from torch import nn
from typing import List, Optional

class Conv(nn.Module):
    """
    A multi‑kernel, batch‑aware convolutional filter with optional depthwise‑separable mode.

    Parameters
    ----------
    kernel_sizes : List[int], optional
        Sizes of the square kernels to apply.  If None, defaults to [2, 3].
    depthwise : bool, optional
        If True, each input channel is convolved separately (depthwise) before a
        point‑wise 1×1 convolution.  This is a lightweight alternative to a full
        multi‑channel convolution.
    threshold : float, optional
        Threshold applied to the sigmoid activation before averaging.  A value of
        0.0 reproduces the behaviour of the original seed.
    bias : bool, optional
        Whether to include a trainable bias term in each convolution.
    """

    def __init__(
        self,
        kernel_sizes: Optional[List[int]] = None,
        depthwise: bool = False,
        threshold: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes or [2, 3]
        self.depthwise = depthwise
        self.threshold = threshold
        self.bias = bias

        # Build a conv layer for each kernel size
        self.convs: nn.ModuleList = nn.ModuleList()
        for k in self.kernel_sizes:
            if self.depthwise:
                # depthwise: in_channels=1, out_channels=1, kernel_size=k
                conv = nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=k,
                    bias=self.bias,
                )
            else:
                # full convolution: in_channels=1, out_channels=1
                conv = nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=k,
                    bias=self.bias,
                )
            self.convs.append(conv)

        # Optional point‑wise 1×1 conv for depthwise mode
        if self.depthwise:
            self.pointwise = nn.Conv2d(
                in_channels=len(self.kernel_sizes),
                out_channels=1,
                kernel_size=1,
                bias=self.bias,
            )
        else:
            self.pointwise = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Activations of shape (B, 1, H', W') where H' and W' depend on the
            kernel sizes and padding.  The activations are sigmoid‑scaled and
            thresholded before the mean is taken over spatial dimensions.
        """
        # List to hold per‑kernel outputs
        outputs = []

        for conv in self.convs:
            out = conv(x)  # (B, 1, H', W')
            out = torch.sigmoid(out - self.threshold)
            outputs.append(out)

        # Combine outputs
        if self.depthwise:
            # Stack along channel dimension and apply point‑wise conv
            stacked = torch.cat(outputs, dim=1)  # (B, K, H', W')
            out = self.pointwise(stacked)  # (B, 1, H', W')
        else:
            # Sum across kernels
            out = torch.stack(outputs, dim=0).sum(dim=0)  # (B, 1, H', W')

        return out

    def calibrate(self, dataloader, device="cpu"):
        """
        Simple calibration routine that scans a range of thresholds on a
        validation set and stores the one that maximises the mean activation.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader that yields (image, label) tuples.
        device : str, optional
            Device on which to run the calibration.
        """
        import numpy as np

        self.to(device)
        best_thr = self.threshold
        best_mean = -np.inf

        for thr in np.linspace(0, 1, 21):
            self.threshold = thr
            means = []
            for imgs, _ in dataloader:
                imgs = imgs.to(device)
                out = self.forward(imgs)
                means.append(out.mean().item())
            mean_val = np.mean(means)
            if mean_val > best_mean:
                best_mean = mean_val
                best_thr = thr

        self.threshold = best_thr
