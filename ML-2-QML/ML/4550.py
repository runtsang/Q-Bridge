"""Hybrid convolutional network combining classical conv, self‑attention, fully connected, and pooling layers."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

# Import subcomponents from seed modules
from Conv import Conv
from SelfAttention import SelfAttention
from QuantumNAT import QFCModel
from QCNN import QCNNModel

class HybridConvNet(nn.Module):
    """Hybrid classical network that emulates a quantum‑inspired architecture.

    Pipeline:
        1. Sliding‑window convolution filter (ConvFilter) extracts patch features.
        2. Classical self‑attention refines the patch representation.
        3. A quantum‑style fully‑connected block (QFCModel) projects to four features.
        4. A QCNN‑style pooling network (QCNNModel) produces a scalar output.
    """

    def __init__(self, kernel_size: int = 2, conv_threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold

        # 1. Convolution filter
        self.conv_filter = Conv()
        self.conv_filter.threshold = conv_threshold

        # 2. Self‑attention
        self.self_attention = SelfAttention()
        self.rotation_params = nn.Parameter(torch.randn(12), requires_grad=True)
        self.entangle_params = nn.Parameter(torch.randn(3), requires_grad=True)

        # 3. Quantum‑style fully‑connected block
        self.qfc = QFCModel()

        # 4. QCNN‑style pooling
        self.qcnn = QCNNModel()

        # Linear layers to map between dimensions
        self.fc_map = nn.Linear(4, 28 * 28)
        self.to_qcnn = nn.Linear(4, 8)

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor of shape (batch, num_patches) where each entry is
        the ConvFilter output for a 2×2 patch."""
        batch, _, h, w = x.shape
        unfold = nn.Unfold(self.kernel_size, stride=self.kernel_size)
        patches = unfold(x)  # (batch, k*k, num_patches)
        patches = patches.permute(0, 2, 1)  # (batch, num_patches, k*k)
        outputs = []
        for i in range(patches.shape[1]):
            patch = patches[:, i, :].reshape(batch, self.kernel_size, self.kernel_size)
            # Convert each patch to numpy and run the classical filter
            conv_out = []
            for b in range(batch):
                conv_out.append(self.conv_filter.run(patch[b].cpu().numpy()))
            outputs.append(conv_out)
        # Shape (batch, num_patches)
        return torch.tensor(outputs, dtype=torch.float32, device=x.device).transpose(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning a probability in [0, 1]."""
        # 1. Extract patch features
        patch_features = self._extract_patches(x)  # (batch, num_patches)

        # Aggregate to a fixed‑size vector (mean over patches)
        vec = patch_features.mean(dim=1)  # (batch,)

        # Expand to 4‑dim vector for self‑attention
        vec4 = vec.unsqueeze(1).repeat(1, 4)  # (batch, 4)

        # 2. Self‑attention
        attn_out = self.self_attention.run(
            self.rotation_params.detach().cpu().numpy(),
            self.entangle_params.detach().cpu().numpy(),
            vec4.detach().cpu().numpy(),
        )
        attn_tensor = torch.tensor(attn_out, dtype=torch.float32, device=x.device)

        # 3. Quantum‑style fully‑connected block
        # Map 4‑dim vector to a 28×28 image
        fc_input = self.fc_map(attn_tensor).view(-1, 1, 28, 28)
        qfc_out = self.qfc(fc_input)  # (batch, 4)

        # 4. QCNN pooling
        qcnn_in = self.to_qcnn(qfc_out)  # (batch, 8)
        qcnn_out = self.qcnn(qcnn_in)    # (batch, 1)

        # Final sigmoid to map to probability
        return torch.sigmoid(qcnn_out)

__all__ = ["HybridConvNet"]
