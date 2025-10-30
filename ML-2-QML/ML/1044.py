"""Enhanced classical convolution with multi‑scale support and learnable threshold.

The class is designed to be a drop‑in replacement for the original Conv filter.
It exposes a small training API (fit, predict) that can be used in a scikit‑learn
pipeline or as a stand‑alone feature extractor.  The implementation uses
PyTorch for tensor operations and includes optional batch‑normalization.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Iterable, Optional


class ConvEnhanced(nn.Module, BaseEstimator, TransformerMixin):
    """A multi‑scale convolutional filter with learnable threshold.

    Parameters
    ----------
    kernel_sizes : Iterable[int]
        Sizes of the kernels to apply (default ``[3, 5]``).  Each size produces
        a single output channel.  The final output is the average of the
        outputs across all scales.
    threshold : float | None, optional
        The threshold used in the sigmoid activation.  If ``None`` the
        threshold is treated as a learnable parameter.
    batch_norm : bool, default=True
        Whether to use a batch‑normalization layer after each convolution.
    bias : bool, default=True
        Whether the bias term in the convolution is set.
    """

    def __init__(
        self,
        kernel_sizes: Iterable[int] = (3, 5),
        threshold: Optional[float] = None,
        batch_norm: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.batch_norm = batch_norm
        self.bias = bias

        # store the conv layers for each scale
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if batch_norm else None

        # threshold handling
        if threshold is None:
            self.threshold = nn.Parameter(torch.zeros(1))
        else:
            self.threshold = torch.tensor([threshold], dtype=torch.float32)

        for k in self.kernel_sizes:
            padding = k // 2  # keep spatial dimensions
            conv = nn.Conv2d(
                1,
                1,
                kernel_size=k,
                padding=padding,
                bias=bias,
            )
            self.convs.append(conv)
            if batch_norm:
                self.bns.append(nn.BatchNorm2d(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the multi‑scale convolutions and return a single channel output.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 1, H, W).
        """
        outputs = []
        for idx, conv in enumerate(self.convs):
            out = conv(x)
            if self.batch_norm:
                out = self.bns[idx](out)
            # apply sigmoid with learnable or fixed threshold
            out = torch.sigmoid(out - self.threshold)
            outputs.append(out)
        # average across scales
        return torch.mean(torch.stack(outputs, dim=0), dim=0)

    def fit(self, X, y=None):
        """Dummy fit method for sklearn compatibility."""
        return self

    def transform(self, X):
        """Apply the filter and return a scalar feature per sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, H, W)
            Input images.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Mean activation per sample.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X.astype("float32"))
        if X.ndim == 3:
            X = X.unsqueeze(1)  # add channel dim
        with torch.no_grad():
            out = self.forward(X)
            # mean over spatial dims
            return out.mean(dim=[1, 2, 3]).cpu().numpy()

    def predict(self, X):
        """Alias for transform."""
        return self.transform(X)


__all__ = ["ConvEnhanced"]
