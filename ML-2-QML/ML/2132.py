import torch
import torch.nn as nn

class QuantumNATEnhanced(nn.Module):
    """Hybrid 2‑D image‑to‑quantum‑feature extractor with learnable patch embedding,
    quantum‑inspired variational branch, and dual heads for classification and
    contrastive projection."""
    def __init__(self, num_classes: int = 4, embed_dim: int = 4,
                 hidden_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        # Patch embedding: 4×4 patches, stride 2, 1-channel input
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(1),  # flatten spatial dims
        )
        # Quantum‑inspired variational branch (classical MLP mimicking amplitude encoding)
        self.quantum_branch = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Classification head (original 4‑class NAT)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.BatchNorm1d(num_classes),
        )
        # Contrastive projection head
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

    def forward(self, x: torch.Tensor):
        # 1. Patch embedding
        x = self.patch_embed(x)          # (B, embed_dim)
        # 2. Variational (classical) processing
        q = self.quantum_branch(x)       # (B, embed_dim)
        # 3. Classification logits
        logits = self.classifier(q)      # (B, num_classes)
        # 4. Contrastive projection
        proj = self.projection(q)        # (B, proj_dim)
        return logits, proj

__all__ = ["QuantumNATEnhanced"]
