import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNHybrid(nn.Module):
    """
    Classical hybrid sampler network.
    Combines a lightweight neural sampler with a plug‑in kernel evaluator.
    The kernel can be swapped with a quantum kernel at runtime.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        # classical sampler
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        # placeholder for kernel (None until set)
        self.kernel = None
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classical sampler.
        """
        return F.softmax(self.sampler(inputs), dim=-1)

    def set_kernel(self, kernel: object) -> None:
        """
        Attach a kernel evaluator (classical or quantum).
        The kernel object must expose a ``matrix(a, b)`` method.
        """
        self.kernel = kernel

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix between two sets of feature vectors.
        Delegates to the attached kernel object.
        """
        if self.kernel is None:
            raise RuntimeError("Kernel not set. Call `set_kernel` first.")
        return self.kernel.matrix(a, b)

__all__ = ["SamplerQNNHybrid"]
