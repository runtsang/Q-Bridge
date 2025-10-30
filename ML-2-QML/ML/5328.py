"""Hybrid kernel module combining classical RBF, optional self‑attention,
fraud‑detection feature extraction, and a sampler network.

It provides the same interface as the original `QuantumKernelMethod`
module, enabling drop‑in replacement for downstream code.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

# Seed implementations
from.QuantumKernelMethod import Kernel
from.SelfAttention import SelfAttention
from.FraudDetection import build_fraud_detection_program, FraudLayerParameters
from.SamplerQNN import SamplerQNN


class HybridKernelMethod(nn.Module):
    """Hybrid kernel that can operate in purely classical mode or
    emulate the quantum behaviour by delegating to the quantum
    equivalents.  The public API matches the seed
    `QuantumKernelMethod` so that downstream code can import it
    without modification.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        use_quantum: bool = False,
        use_attention: bool = False,
        use_fraud: bool = False,
        use_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_quantum = use_quantum
        self.use_attention = use_attention
        self.use_fraud = use_fraud
        self.use_sampler = use_sampler

        # Classical RBF kernel
        self.rbf = Kernel(gamma)

        # Quantum kernel placeholder – kept as classical for compatibility.
        self.qkernel = Kernel() if use_quantum else None

        # Self‑attention block
        self.attention = SelfAttention() if use_attention else None

        # Fraud‑detection feature extractor
        self.fraud_net = (
            build_fraud_detection_program(
                FraudLayerParameters(
                    bs_theta=0.0,
                    bs_phi=0.0,
                    phases=(0.0, 0.0),
                    squeeze_r=(0.0, 0.0),
                    squeeze_phi=(0.0, 0.0),
                    displacement_r=(0.0, 0.0),
                    displacement_phi=(0.0, 0.0),
                    kerr=(0.0, 0.0),
                ),
                [],
            )
            if use_fraud
            else None
        )

        # Sampler QNN
        self.sampler = SamplerQNN() if use_sampler else None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return a Gram matrix between two batches of samples."""
        if self.use_fraud and self.fraud_net is not None:
            x = self.fraud_net(x)
            y = self.fraud_net(y)

        if self.use_attention and self.attention is not None:
            rot = np.random.rand(12)
            ent = np.random.rand(3)
            x = self.attention.run(rot, ent, x.detach().cpu().numpy())
            y = self.attention.run(rot, ent, y.detach().cpu().numpy())
            x = torch.as_tensor(x, dtype=torch.float32)
            y = torch.as_tensor(y, dtype=torch.float32)

        if self.use_quantum and self.qkernel is not None:
            k = self.qkernel(x, y)
        else:
            k = self.rbf(x, y)

        if self.use_sampler and self.sampler is not None:
            probs = self.sampler(x)
            k = k * probs.sum(dim=-1, keepdim=True)

        return k

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Convenience wrapper that mirrors the seed function."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernelMethod"]
