import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridAttentionQuanvolution(nn.Module):
    """
    Classical hybrid module combining a 2‑D convolutional feature extractor
    (inspired by the quanvolution filter) with a self‑attention block.
    The interface mimics the quantum version so that the same experiment
    scripts can be swapped between the two implementations.
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        # 2x2 patch extraction via Conv2d with stride 2 (28x28 -> 14x14 patches)
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        # Linear layers to produce query, key, value
        self.query_proj = nn.Linear(4, embed_dim, bias=False)
        self.key_proj   = nn.Linear(4, embed_dim, bias=False)
        self.value_proj = nn.Linear(4, embed_dim, bias=False)
        # Final linear head
        self.classifier = nn.Linear(embed_dim * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (B,1,28,28)
        patches = self.conv(x)  # (B,4,14,14)
        B, C, H, W = patches.shape
        patches = patches.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        # Compute query/key/value
        Q = self.query_proj(patches)  # (B,N,embed)
        K = self.key_proj(patches)
        V = self.value_proj(patches)
        scores = torch.softmax((Q @ K.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        attn_out = scores @ V  # (B,N,embed)
        attn_out = attn_out.permute(0, 2, 1).contiguous().view(B, -1)
        logits = self.classifier(attn_out)
        return F.log_softmax(logits, dim=-1)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Compatibility wrapper: rotation_params and entangle_params are
        interpreted as weight matrices for the query and key projections.
        """
        rot = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        ent = torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        with torch.no_grad():
            self.query_proj.weight.copy_(rot)
            self.key_proj.weight.copy_(ent)
        out = self.forward(torch.as_tensor(inputs, dtype=torch.float32))
        return out.detach().cpu().numpy()

__all__ = ["HybridAttentionQuanvolution"]
