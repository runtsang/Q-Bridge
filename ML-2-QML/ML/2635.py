import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalPatchExtractor(nn.Module):
    """Classical approximation of a 2×2 quantum patch extractor.
    Maps each 2×2 patch to a 4‑dim feature vector via a small neural network.
    """
    def __init__(self, patch_size: int = 2, out_features: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.out_features = out_features
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(patch_size * patch_size, 8),
            nn.ReLU(),
            nn.Linear(8, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "Image dimensions must be divisible by patch_size"
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches shape: (B, C, H//ps, W//ps, ps, ps)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        # combine channel and patch dims
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(B, -1, C * self.patch_size * self.patch_size)
        encoded = self.encoder(patches)  # (B, num_patches, out_features)
        return encoded.view(B, -1)

class ClassicalHybridHead(nn.Module):
    """Classical head that mimics a quantum expectation layer using a sigmoid."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        probs = torch.sigmoid(logits + self.shift)
        return probs

class UnifiedQuanvolutionHybrid(nn.Module):
    """Hybrid model that uses a classical patch extractor and a classical hybrid head.
    This is the classical counterpart to the quantum implementation.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.patch_extractor = ClassicalPatchExtractor(patch_size=2, out_features=4)
        # For 28×28 images: 14×14 patches → 196 patches, each 4‑dim → 784 features
        self.classifier = nn.Linear(784, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.patch_extractor(x)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["UnifiedQuanvolutionHybrid"]
