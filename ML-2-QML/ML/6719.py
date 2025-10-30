"""Classical QuanvolutionHybrid with learnable patch extraction.

This model implements a dual‑mode interface: a purely classical branch that uses a
learnable 2‑D convolution as a patch extractor followed by a linear classifier,
and a placeholder for a hybrid quantum‑classical branch that can be activated
by passing mode='hybrid'. The design keeps the public API identical to the
original seed while adding a learnable patch extractor and a clean interface
for future quantum extensions.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """Hybrid quanvolution model with learnable patch extractor.

    Parameters
    ----------
    patch_size : int, default 2
        Size of the square patch extracted by the convolution.
    stride : int, default 2
        Stride of the convolution, controlling the number of patches.
    num_classes : int, default 10
        Number of output classes.
    mode : str, default 'classical'
        Default mode for the forward pass; must be either 'classical' or
        'hybrid'. The hybrid mode is a placeholder in the classical
        implementation.
    """
    def __init__(
        self,
        patch_size: int = 2,
        stride: int = 2,
        num_classes: int = 10,
        mode: str = "classical",
    ) -> None:
        super().__init__()
        self._mode = mode
        self.patch_size = patch_size
        self.stride = stride
        self.num_classes = num_classes

        # Learnable patch extractor: 1 input channel → 4 output channels
        self.patch_extractor = nn.Conv2d(
            1, 4, kernel_size=patch_size, stride=stride
        )

        # Linear classifier
        num_patches = (28 // stride) ** 2
        self.linear = nn.Linear(4 * num_patches, num_classes)

    def forward(self, x: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, 28, 28).
        mode : str, optional
            Override the instance mode for this call.  Must be
            'classical' or 'hybrid'.

        Returns
        -------
        torch.Tensor
            Log‑softmax of logits.
        """
        if mode is None:
            mode = self._mode

        if mode == "classical":
            patches = self.patch_extractor(x)  # (bsz, 4, H', W')
            features = patches.view(patches.size(0), -1)  # flatten
            logits = self.linear(features)
            return F.log_softmax(logits, dim=-1)
        else:
            # Hybrid mode is not implemented in the classical module
            raise NotImplementedError(
                "Hybrid mode is not supported in the classical implementation"
            )

__all__ = ["QuanvolutionHybrid"]
