import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATExtended(nn.Module):
    """A hybrid model that combines a 2‑D convolutional backbone, a multi‑head attention
    projection, and a final fully‑connected head. The attention module allows the
    network to focus on spatial regions before the linear projection, improving
    representation quality for downstream tasks.

    The network outputs a 4‑dimensional vector, matching the original seed, but now
    contains an additional learnable projection layer that can be fine‑tuned or
    used in contrastive learning pipelines.
    """

    def __init__(self, in_ch: int = 1, num_heads: int = 4, head_dim: int = 16) -> None:
        super().__init__()
        # Convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Multi‑head attention projection
        self.attention = nn.MultiheadAttention(embed_dim=16 * 7 * 7,
                                               num_heads=num_heads,
                                               batch_first=True)
        # Linear projection head
        self.proj = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.backbone(x)  # -> (bsz, 16, 7, 7)
        flattened = features.view(bsz, -1)  # -> (bsz, 16*7*7)
        # Multi‑head attention expects seq_len, batch, embed_dim
        attn_out, _ = self.attention(flattened.unsqueeze(1),  # query
                                     flattened.unsqueeze(1),  # key
                                     flattened.unsqueeze(1))  # value
        attn_out = attn_out.squeeze(1)  # -> (bsz, 16*7*7)
        out = self.proj(attn_out)
        return self.norm(out)

__all__ = ["QuantumNATExtended"]
