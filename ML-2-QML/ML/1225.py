import torch
import torch.nn as nn
import torch.nn.functional as F

class Quanvolution__gen168(nn.Module):
    """
    Classical hybrid filter with attention and multi‑head self‑attention.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, patch_size: int = 2) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=patch_size, stride=patch_size)
        self.attn = nn.Sequential(
            nn.Linear(4, 4),
            nn.Sigmoid()
        )
        self.token_proj = nn.Linear(4, 128)
        self.backbone = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.mha = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract 2×2 patches via convolution
        patches = self.conv(x)  # (B, 4, H', W')
        B, C, H, W = patches.shape
        patches = patches.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, N, 4)
        # Attention gating
        attn_weights = self.attn(patches)  # (B, N, 4)
        gated = patches * attn_weights
        # Project to embedding space
        token_emb = self.token_proj(gated)  # (B, N, 128)
        # Multi‑head self‑attention
        attn_output, _ = self.mha(token_emb, token_emb, token_emb)  # (B, N, 128)
        # Aggregate tokens
        pooled = attn_output.mean(dim=1)  # (B, 128)
        # Backbone and classification
        features = self.backbone(pooled)  # (B, 128)
        logits = self.classifier(features)  # (B, num_classes)
        return F.log_softmax(logits, dim=-1)

__all__ = ["Quanvolution__gen168"]
