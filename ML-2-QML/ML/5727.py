import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionGen112(nn.Module):
    """
    Classical hybrid architecture that emulates a quantum kernel for each 2×2 patch
    using a trainable linear mapping, then fuses the encoded patches with a
    classical convolutional layer before a fully‑connected classifier.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 patch_size: int = 2, conv_out_channels: int = 16):
        super().__init__()
        self.patch_size = patch_size
        # Extract 2×2 patches and map to 4‑dimensional feature vectors
        self.patch_conv = nn.Conv2d(in_channels, 4, kernel_size=patch_size,
                                    stride=patch_size)
        # Classical approximation of the quantum kernel
        self.quantum_approx = nn.Linear(4, 4)
        # Classical convolution to fuse patch features
        self.classical_conv = nn.Conv2d(4, conv_out_channels,
                                        kernel_size=3, padding=1)
        # Final classifier
        self.fc = nn.Linear(conv_out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        patches = self.patch_conv(x)  # (batch, 4, 14, 14)
        # Flatten spatial dimensions for linear layer
        patches_flat = patches.permute(0, 2, 3, 1).reshape(-1, 4)
        encoded = self.quantum_approx(patches_flat)  # (batch*14*14, 4)
        # Reshape back to image‑like tensor
        encoded = encoded.view(patches.size(0), 14, 14, 4).permute(0, 3, 1, 2)
        fused = self.classical_conv(encoded)  # (batch, conv_out_channels, 14, 14)
        flattened = fused.view(fused.size(0), -1)
        logits = self.fc(flattened)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen112"]
