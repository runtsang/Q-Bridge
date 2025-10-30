import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumNATHybrid(nn.Module):
    """A hybrid classical model inspired by Quantum-NAT.
    Extends the original CNN + FC architecture with a transformer encoder
    and a contrastive projection head to learn richer, domain‑agnostic
    representations. The architecture is fully compatible with the
    original input shape (batch, 1, 28, 28)."""

    def __init__(self, num_classes: int = 4, dim: int = 64, nhead: int = 4,
                 num_layers: int = 2, proj_dim: int = 32, dropout: float = 0.1):
        super().__init__()

        # CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )

        # Flattened feature dimension: 16 * 7 * 7 = 784
        self.feature_dim = 16 * 7 * 7

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=nhead, dim_feedforward=dim, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection head for contrastive learning
        self.proj_head = nn.Sequential(
            nn.Linear(self.feature_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )

        # Normalisation
        self.norm = nn.BatchNorm1d(self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for classification."""
        bsz = x.size(0)
        # CNN feature extraction
        feats = self.features(x)          # (bsz, 16, 7, 7)
        feats = feats.view(bsz, -1)       # (bsz, 784)

        # Transformer expects (seq_len, batch, d_model)
        seq = feats.unsqueeze(0)          # (1, bsz, 784)
        transformed = self.transformer(seq)  # (1, bsz, 784)
        transformed = transformed.squeeze(0)  # (bsz, 784)

        # Normalise
        out = self.norm(transformed)

        # Classification logits
        logits = self.classifier(out)
        return logits

    def contrastive_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning contrastive embeddings."""
        bsz = x.size(0)
        feats = self.features(x).view(bsz, -1)
        seq = feats.unsqueeze(0)
        transformed = self.transformer(seq).squeeze(0)
        out = self.norm(transformed)
        proj = self.proj_head(out)
        return proj

    def nt_xent_loss(self, z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """Normalized temperature-scaled cross entropy loss (NT-Xent)."""
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # 2n
        z = F.normalize(z, dim=1)

        similarity_matrix = torch.mm(z, z.t())  # (2n, 2n)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

        positives = torch.cat([torch.diag(similarity_matrix, batch_size),
                               torch.diag(similarity_matrix, -batch_size)], dim=0)

        logits = similarity_matrix / temperature
        labels = torch.arange(batch_size, device=z.device).repeat(2)
        loss = F.cross_entropy(logits, labels)
        return loss

__all__ = ["QuantumNATHybrid"]
