import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    """
    Depth‑wise separable convolution: depth‑wise followed by point‑wise.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))

class MultiHeadQuanvolutionFilter(nn.Module):
    """
    Classical‑to‑quantum mapping that extracts 2×2 patches and runs them through
    a depth‑wise separable convolution with multiple heads.  Each head produces
    a 4‑dimensional vector that will be fed to the quantum module.
    """
    def __init__(self,
                 patch_size: int = 2,
                 in_channels: int = 1,
                 head_dim: int = 4,
                 num_heads: int = 3,
                 kernel_size: int = 2,
                 stride: int = 2) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Depth‑wise separable conv per head
        self.heads = nn.ModuleList([
            SeparableConv2d(in_channels, head_dim, kernel_size, stride)
            for _ in range(num_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Concatenated head outputs of shape (B, num_heads * 4, H_out, W_out)
        """
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(x))          # (B, 4, H_out, W_out)
        return torch.cat(head_outputs, dim=1)      # (B, H*4, H_out, W_out)

class QuanvolutionGen357Classifier(nn.Module):
    """
    Hybrid network that applies the multi‑head separable quanvolution filter
    followed by a variational quantum layer and a linear classifier.
    """
    def __init__(self,
                 num_classes: int = 10,
                 patch_size: int = 2,
                 in_channels: int = 1,
                 head_dim: int = 4,
                 num_heads: int = 3) -> None:
        super().__init__()
        self.qfilter = MultiHeadQuanvolutionFilter(
            patch_size=patch_size,
            in_channels=in_channels,
            head_dim=head_dim,
            num_heads=num_heads
        )
        self.quantum_layer = None  # to be set in the QML subclass
        # Linear head expects 4 * 14 * 14 * num_heads features (MNIST 28×28)
        self.linear = nn.Linear(head_dim * 14 * 14 * num_heads, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classical filter, quantum layer, and linear head.
        """
        features = self.qfilter(x)                     # (B, H*4, 14, 14)
        # Reshape to (B, H*14*14, 4) for quantum processing
        B, H, H_out, W_out = features.shape
        features = features.view(B, H * H_out * W_out, 4)
        # Quantum processing
        q_features = self.quantum_layer(features)      # (B, H*14*14, 4)
        q_features = q_features.view(B, -1)            # (B, 4*H*14*14)
        logits = self.linear(q_features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["SeparableConv2d", "MultiHeadQuanvolutionFilter",
           "QuanvolutionGen357Classifier"]
