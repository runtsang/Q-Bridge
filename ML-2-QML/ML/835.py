import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """
    Classical CNN with a multi‑head self‑attention block and a linear projection.
    """
    def __init__(self, in_channels: int = 1, num_heads: int = 4, attn_dim: int = 28, hidden_dim: int = 128):
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Attention block
        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)
        # Linear projection that receives both flattened features and attention output
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7 + attn_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)                # (B, 16, 7, 7)
        flat = feat.view(bsz, -1)              # (B, 784)
        # Prepare sequence for attention
        seq_len = flat.shape[1] // self.attn.embed_dim
        attn_input = flat[:, :seq_len * self.attn.embed_dim].view(bsz, seq_len, self.attn.embed_dim)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        attn_out = attn_output.mean(dim=1)      # (B, attn_dim)
        # Concatenate and project
        combined = torch.cat([flat, attn_out], dim=1)
        out = self.fc(combined)
        return self.norm(out)
