"""Hybrid classical‑quantum convolution module.

Combines a PyTorch Conv2d layer with a Qiskit‑based quantum patch filter.
The design integrates concepts from the two seed implementations:
- Classical Conv (kernel, threshold) and QuanvolutionFilter (multi‑channel conv).
- Quantum filter (random circuit, measurement) and QuanvolutionFilter (quantum kernel).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from.qml_module import QuantumPatchFilter  # quantum filter implementation


class HybridConv(nn.Module):
    """Classical‑Quantum hybrid convolutional filter.

    Parameters
    ----------
    kernel_size : int
        Size of the square filter (default 2).
    stride : int
        Stride of the convolution (default 2).
    in_channels : int
        Number of input channels (default 1).
    out_channels : int
        Number of classical convolutional output channels (default 4).
    quantum_depth : int
        Depth of the random quantum circuit applied to each patch
        (default 2).
    backend : str
        Qiskit simulator backend name (default 'qasm_simulator').
    shots : int
        Number of shots for quantum measurement (default 100).
    threshold : float
        Threshold for encoding pixel values into rotation angles
        (default 127).
    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 2,
        in_channels: int = 1,
        out_channels: int = 4,
        quantum_depth: int = 2,
        backend: str = "qasm_simulator",
        shots: int = 100,
        threshold: float = 127,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.quantum_filter = QuantumPatchFilter(
            kernel_size=kernel_size,
            backend=backend,
            shots=shots,
            threshold=threshold,
            depth=quantum_depth,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hybrid filter to input image.

        The classical convolution is applied first, producing a feature map.
        A quantum filter is then applied to each patch of the input image.
        The two representations are concatenated along the channel dimension
        and flattened to produce a single feature vector per sample.
        """
        # Classical convolution
        conv_out = self.conv(x)  # (batch, out_channels, H', W')
        conv_flat = conv_out.view(conv_out.size(0), -1)

        # Quantum patch filter
        patches = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # patches shape: (batch, in_channels, num_patches_h, num_patches_w, k, k)
        batch, ch, ph, pw, kx, ky = patches.shape
        patches = patches.contiguous().view(-1, kx, ky)  # (batch*ch*ph*pw, k, k)

        q_vals = torch.tensor(
            [self.quantum_filter.run(patch.numpy()) for patch in patches]
        ).to(x.device)  # (batch*ch*ph*pw,)

        q_vals = q_vals.view(batch, ch, ph * pw)  # (batch, ch, num_patches)
        q_features = q_vals.mean(dim=1)  # (batch, num_patches)
        q_flat = q_features.view(batch, -1)

        return torch.cat([conv_flat, q_flat], dim=1)


__all__ = ["HybridConv"]
