import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEncoder(nn.Module):
    """Classical 2×2 patch extractor using a convolution."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # (B, 4, 14, 14)

class PerPatchMLP(nn.Module):
    """Per‑patch MLP that projects 4‑dimensional patch features to a latent space."""
    def __init__(self, patch_dim: int = 4, latent_dim: int = 32) -> None:
        super().__init__()
        self.linear = nn.Linear(patch_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # (B, seq_len, latent_dim)

class ClassicalTransformerClassifier(nn.Module):
    """Transformer classifier that operates on a sequence of latent vectors."""
    def __init__(self, seq_len: int, embed_dim: int, num_heads: int,
                 num_layers: int, num_classes: int) -> None:
        super().__init__()
        self.positional = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.positional
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

class QuanvolutionAutoTransformer(nn.Module):
    """Hybrid classical model that mirrors the quantum‑enhanced architecture."""
    def __init__(self, latent_dim: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        self.patch_encoder = PatchEncoder()
        self.per_patch_mlp = PerPatchMLP(patch_dim=4, latent_dim=latent_dim)
        self.transformer = ClassicalTransformerClassifier(
            seq_len=14 * 14,
            embed_dim=latent_dim,
            num_heads=4,
            num_layers=2,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract 2×2 patches
        patches = self.patch_encoder(x)          # (B, 4, 14, 14)
        patches = patches.permute(0, 2, 3, 1).contiguous()  # (B, 14, 14, 4)
        patches = patches.view(x.size(0), -1, 4)             # (B, 196, 4)
        # Project patches to latent space
        latent = self.per_patch_mlp(patches)                 # (B, 196, latent_dim)
        # Classify with transformer
        logits = self.transformer(latent)
        return logits

__all__ = ["QuanvolutionAutoTransformer"]
