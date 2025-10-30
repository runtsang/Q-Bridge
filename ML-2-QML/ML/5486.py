import torch
import torch.nn as nn
import numpy as np

# Import the auxiliary modules from the seed repository
from.Conv import Conv
from.QuantumNAT import QFCModel
from.QCNN import QCNNModel
from.QuantumKernelMethod import Kernel

class HybridNAT(nn.Module):
    """
    Classical hybrid model that fuses ideas from the four reference pairs.

    Architecture
    ------------
    * **ConvFilter** – 2×2 convolution emulating a quanvolution layer.
    * **QFCModel** – classical CNN followed by a fully‑connected projection
      (from Quantum‑NAT).
    * **QCNNModel** – stack of fully‑connected layers that mirrors the
      QCNN helper.
    * **Kernel** – radial‑basis‑function kernel for similarity scoring.

    The forward pass concatenates the scalar output of the ConvFilter
    with the 4‑dimensional QFC output and the 1‑dimensional QCNN output.
    A helper ``kernel_score`` can be used to evaluate the RBF kernel
    between two batches of data.
    """

    def __init__(self, kernel_gamma: float = 1.0) -> None:
        super().__init__()
        self.conv_filter = Conv()
        self.qfc = QFCModel()
        self.qcnn = QCNNModel()
        self.kernel = Kernel(gamma=kernel_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv filter produces a scalar per sample (non‑differentiable)
        conv_scalar = torch.tensor(
            self.conv_filter.run(x[0, 0].cpu().numpy())
        ).unsqueeze(0)
        conv_feat = conv_scalar.expand(x.shape[0], 1)

        # Classical CNN + FC features
        qfc_feat = self.qfc(x)          # (batch, 4)

        # QCNN output
        qcnn_out = self.qcnn(x)         # (batch, 1)

        # Concatenate along the feature dimension
        return torch.cat([conv_feat, qfc_feat, qcnn_out], dim=-1)

    def kernel_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return the RBF kernel matrix between two batches.
        """
        return self.kernel(x, y)
