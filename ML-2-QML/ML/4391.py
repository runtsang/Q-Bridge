import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridSamplerQNN(nn.Module):
    """
    Classical emulation of a quantum sampler.
    Combines a QCNN‑style feature extractor, a self‑attention block
    and a fully‑connected layer that mimics a quantum sampler.
    """
    def __init__(self, input_dim: int = 2, embed_dim: int = 4):
        super().__init__()
        # QCNN‑style feature extractor
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, 8), nn.Tanh(),
            nn.Linear(8, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh()
        )
        # Self‑attention (single head for simplicity)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                               num_heads=1,
                                               batch_first=True)
        # Fully connected sampler
        self.sampler = nn.Linear(embed_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        feat = self.feature_map(x)
        # Prepare sequence for attention
        seq = feat.unsqueeze(1)  # (batch, seq_len=1, embed_dim)
        attn_out, _ = self.attention(seq, seq, seq)
        # Sampler output
        logits = self.sampler(attn_out.squeeze(1))
        return F.softmax(logits, dim=-1)

def SamplerQNN() -> HybridSamplerQNN:
    """
    Factory function mirroring the original API.
    Returns an instance of the hybrid classical sampler.
    """
    return HybridSamplerQNN()

__all__ = ["HybridSamplerQNN", "SamplerQNN"]
