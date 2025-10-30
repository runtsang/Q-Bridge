import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Callable, Iterable, List, Optional

__all__ = ["ConvGenHybrid"]

class ConvFilter(nn.Module):
    """Classical 2‑D convolution that emulates the original Quanv filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        return torch.sigmoid(logits - self.threshold)

class FullyConnectedLayer(nn.Module):
    """Classical fully‑connected layer mirroring the FCL example."""
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        vals = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(vals)).mean(dim=0)

class HybridAttention(nn.Module):
    """Hybrid multi‑head attention that optionally routes each head through a quantum circuit."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 use_quantum: bool = False,
                 q_circuits: Optional[List[Callable]] = None,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_quantum = use_quantum

        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)

        self.q_heads = None
        if use_quantum and q_circuits is not None:
            self.q_heads = [fn() for fn in q_circuits]

    def set_quantum(self, enabled: bool):
        self.use_quantum = enabled

    def _quantum_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run each head through its quantum circuit and return a tensor of shape
        (batch, seq, embed_dim). The quantum circuits are expected to accept a
        1‑D tensor of shape (seq, d_k) and return a tensor of the same shape."""
        batch, seq, _ = x.shape
        outputs = []
        for h, q_head in enumerate(self.q_heads):
            head_out = q_head(x[:, :, h * self.d_k : (h + 1) * self.d_k])
            outputs.append(head_out)
        return torch.stack(outputs, dim=2).reshape(batch, seq, self.embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        batch, seq, _ = x.shape
        q = q.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = self.dropout(scores.masked_fill(mask == 0, -1e9))
        else:
            scores = self.dropout(scores)

        attn_out = torch.matmul(scores, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)

        if self.use_quantum and self.q_heads is not None:
            q_out = self._quantum_forward(x)
            attn_out = 0.5 * attn_out + 0.5 * q_out
        return attn_out

class HybridFeedForward(nn.Module):
    """Hybrid feed‑forward that can be classical or quantum."""
    def __init__(self, embed_dim: int, ffn_dim: int, use_quantum: bool = False):
        super().__init__()
        self.use_quantum = use_quantum
        self.dropout = nn.Dropout(0.1)
        if use_quantum:
            self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
            self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        else:
            self.linear1 = nn.Linear(embed_dim, ffn_dim)
            self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def set_quantum(self, enabled: bool):
        self.use_quantum = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            out = self.linear1(x)
            out = torch.sigmoid(out)
            return self.linear2(out)
        else:
            return self.linear2(F.relu(self.linear1(x)))

class TransformerHybridBlock(nn.Module):
    """A single transformer block that may contain quantum attention / feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 use_quantum_attention: bool = False,
                 use_quantum_ffn: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = HybridAttention(embed_dim,
                                    num_heads,
                                    use_quantum=use_quantum_attention,
                                    q_circuits=[lambda: None] * use_quantum_attention,
                                    dropout=dropout)
        self.ffn = HybridFeedForward(embed_dim, ffn_dim, use_quantum=use_quantum_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.attn.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.ffn.dropout(ffn_out))

class ConvGenHybrid(nn.Module):
    """Hybrid model combining a convolution, a fully‑connected layer, and a stack of transformer blocks.
    The attention or feed‑forward sub‑modules can be toggled to use quantum circuits."""
    def __init__(self,
                 embed_dim: int = 32,
                 num_heads: int = 4,
                 ffn_dim: int = 64,
                 num_blocks: int = 2,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 num_features: int = 1,
                 use_quantum_attention: bool = False,
                 use_quantum_ffn: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.conv = ConvFilter(kernel_size=kernel_size, threshold=threshold)
        self.fcl = FullyConnectedLayer(n_features=num_features)
        self.blocks = nn.ModuleList([
            TransformerHybridBlock(embed_dim,
                                   num_heads,
                                   ffn_dim,
                                   use_quantum_attention,
                                   use_quantum_ffn,
                                   dropout)
            for _ in range(num_blocks)
        ])
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be a batch of images (B, 1, H, W)
        x = self.conv(x)          # (B, 1, k, k)
        # flatten the convolution output to a sequence
        B, C, H, W = x.shape
        seq = H * W
        x = x.view(B, seq, C)      # (B, seq, 1)
        # pass through fully‑connected layer to expand feature dimension
        thetas = [float(i) for i in range(seq)]  # dummy theta sequence
        fcl_out = self.fcl(thetas)              # (1,)
        x = x + fcl_out  # broadcast to sequence
        for blk in self.blocks:
            x = blk(x)
        # pool over sequence
        x = x.mean(dim=1)
        return self.classifier(x)

    def load_quantum(self, enabled: bool = True):
        """Toggle quantum sub‑modules inside the transformer blocks."""
        for blk in self.blocks:
            blk.attn.set_quantum(enabled)
            blk.ffn.set_quantum(enabled)

    def freeze_classical(self):
        """Freeze all classical parameters so only quantum weights are trainable."""
        for param in self.parameters():
            param.requires_grad = False
        for blk in self.blocks:
            if not blk.attn.use_quantum:
                for param in blk.attn.parameters():
                    param.requires_grad = True
            if not blk.ffn.use_quantum:
                for param in blk.ffn.parameters():
                    param.requires_grad = True
