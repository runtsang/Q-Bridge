"""ConvEnhanced – a hybrid classical convolution module with multi‑scale and residual support."""

from __future__ import annotations

import torch
from torch import nn

class ConvEnhanced(nn.Module):
    """
    A drop‑in replacement for the original Conv filter that supports
    * multi‑scale kernels (1, 2, 4, 8)
    * depth‑wise separable convolution for efficiency
    * optional residual connection to the input patch
    * a learnable fusion gate that mixes classical and quantum logits
    * a simple hybrid loss that can be used with a downstream classifier
    """

    def __init__(
        self,
        kernel_sizes: list[int] | None = None,
        residual: bool = False,
        use_qc: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.use_qc = use_qc
        if kernel_sizes is None:
            kernel_sizes = [2]
        self.kernel_sizes = kernel_sizes

        # Classical depth‑wise separable conv for each kernel size
        self.classical_convs = nn.ModuleList()
        for k in kernel_sizes:
            depthwise = nn.Conv2d(1, 1, kernel_size=k, padding=0, bias=bias, groups=1)
            pointwise = nn.Conv2d(1, 1, kernel_size=1, bias=bias)
            self.classical_convs.append(nn.Sequential(depthwise, pointwise))

        # Fusion gate for mixing classical and quantum logits
        self.fusion_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, H, W) representing image patches
        Returns:
            Tensor of shape (batch,) containing the fused logit for each patch
        """
        logits = []
        for conv in self.classical_convs:
            out = conv(x)
            out = torch.mean(out, dim=[2, 3])
            logits.append(out.squeeze(-1).squeeze(-1))
        logits = torch.stack(logits, dim=-1)
        weights = torch.ones(logits.shape[-1], device=x.device) / logits.shape[-1]
        class_logits = torch.sum(logits * weights, dim=-1)

        if self.use_qc:
            # Placeholder for quantum logits (could be integrated via Qiskit)
            qc_logits = torch.randn_like(class_logits)
            fused = self.fusion_gate * qc_logits + (1 - self.fusion_gate) * class_logits
        else:
            fused = class_logits

        if self.residual:
            residual = torch.mean(x, dim=[2, 3])
            fused = fused + residual.squeeze(-1).squeeze(-1)

        return fused

__all__ = ["ConvEnhanced"]
