import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuanvolution(nn.Module):
    """
    Classical implementation of a hybrid quanvolution filter.
    Applies a 2×2 convolution, optional thresholded sigmoid activation,
    and an optional quantum‑inspired orthogonal transformation before
    a linear classifier.  The design mirrors the quantum filter
    but remains fully classical for efficient training.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 use_quantum_kernel: bool = False, seed: int | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # 4 output channels to match the quantum filter's 4‑dimensional patch
        self.conv = nn.Conv2d(1, 4, kernel_size=kernel_size, stride=2, bias=True)
        self.use_quantum_kernel = use_quantum_kernel
        if self.use_quantum_kernel:
            torch.manual_seed(seed)
            # Create a fixed orthogonal matrix to emulate a quantum kernel
            q, _ = torch.qr(torch.randn(4, 4))
            self.register_buffer("q_kernel", q)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of 28×28 grayscale images.
        """
        # Convolutional feature extraction
        feat = self.conv(x)  # (B, 4, 14, 14)
        # Thresholded sigmoid activation
        feat = torch.sigmoid(feat - self.threshold)
        # Optional quantum‑inspired orthogonal transform
        if self.use_quantum_kernel:
            B, C, H, W = feat.shape
            feat = feat.view(B, C, -1)  # (B, 4, H*W)
            feat = torch.einsum("ij,bjk->bik", self.q_kernel, feat)
            feat = feat.view(B, -1)
        else:
            feat = feat.view(feat.size(0), -1)
        logits = self.linear(feat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolution"]
