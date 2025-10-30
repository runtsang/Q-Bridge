import torch
from torch import nn
import torch.nn.functional as F

class QCNNQuanvolutionHybrid(nn.Module):
    """Hybrid classical model that mimics a QCNN with a patch‑wise quantum‑kernel inspired filter.

    The architecture first extracts 2×2 patches from a single‑channel image using a
    learnable Conv2d, then applies a lightweight random linear “quantum kernel” to
    each patch (the same idea as the Quanvolution filter).  The flattened
    representation is fed into a stack of fully‑connected layers that follow the
    topology of the QCNNModel from the seed.  This design preserves the
    convolution‑pooling intuition of the QCNN while allowing a classical
    approximation of the quantum kernel, enabling efficient training on a CPU.
    """

    def __init__(self, num_channels: int = 1, patch_size: int = 2,
                 image_size: int = 28, num_classes: int = 10) -> None:
        super().__init__()
        # Patch extraction – identical to QuanvolutionFilter
        self.patch_conv = nn.Conv2d(num_channels, 4, kernel_size=patch_size,
                                    stride=patch_size, bias=False)

        # Random linear layer per patch to emulate a quantum kernel
        # The number of patches is (image_size//patch_size)^2
        num_patches = (image_size // patch_size) ** 2
        self.patch_kernel = nn.Linear(4, 4, bias=False)
        nn.init.kaiming_normal_(self.patch_kernel.weight, nonlinearity='linear')

        # Fully‑connected stack mirroring QCNNModel
        self.fc = nn.Sequential(
            nn.Linear(4 * num_patches, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Extract patches
        patches = self.patch_conv(x)  # shape: (B, 4, H', W')
        # Flatten patches to (B, num_patches, 4)
        patches = patches.view(patches.size(0), 4, -1).transpose(1, 2)
        # Apply random kernel to each patch
        patches = self.patch_kernel(patches)  # (B, num_patches, 4)
        # Flatten all patch features
        features = patches.reshape(patches.size(0), -1)
        # Forward through the QCNN‑like head
        logits = self.fc(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QCNNQuanvolutionHybrid"]
