import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ConvGen444(nn.Module):
    """
    Classical convolutional filter with optional quantum augmentation.
    The module can be used as a drop‑in replacement for the original
    Conv filter in the anchor file.  When ``use_quantum`` is true the
    module forwards the convolution output through a small variational
    quantum circuit that is differentiable via a custom autograd
    function.  The quantum part can be swapped out for any
    Qiskit/Braket circuit without changing the interface.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 stride: int = 1,
                 bias: bool = True,
                 threshold: float = 0.0,
                 use_quantum: bool = False,
                 quantum_module: Optional[nn.Module] = None,
                 ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.threshold = threshold
        self.use_quantum = use_quantum

        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size,
                              stride=stride, bias=bias)

        if use_quantum:
            if quantum_module is None:
                raise ValueError("quantum_module must be provided when use_quantum=True")
            self.quantum_module = quantum_module
        else:
            self.quantum_module = None

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        data : torch.Tensor
            Input image of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Scalar activation for each batch element.
        """
        logits = self.conv(data)
        # apply sigmoid with threshold
        act = torch.sigmoid(logits - self.threshold)

        if self.use_quantum:
            # flatten spatial dims per batch
            flat = act.view(act.size(0), -1)
            # quantum module expects 1‑D tensor per sample
            q_out = self.quantum_module(flat)
            # combine classically
            out = (act.mean(dim=(1,2,3)) + q_out).mean()
        else:
            out = act.mean(dim=(1,2,3))
        return out

__all__ = ["ConvGen444"]
