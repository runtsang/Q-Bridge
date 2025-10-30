import torch
from torch import nn
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block used by :class:`HybridConv`."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class QCNNBlock(nn.Module):
    """A tiny QCNN‑style feed‑forward block mirroring the seed."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 4), nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.layers(x))

class HybridConv(nn.Module):
    """Hybrid classical convolutional layer that combines a traditional
    convolution, a QCNN‑style dense block, a self‑attention module and
    fraud‑detection style scaling."""
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        use_attention: bool = False,
        attention_dim: int = 1,
        clip_weights: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.qcnn = QCNNBlock()
        self.use_attention = use_attention
        if use_attention:
            self.attention = ClassicalSelfAttention(attention_dim)
        self.clip_weights = clip_weights
        # Fraud‑style scaling parameters
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))
        if clip_weights:
            self._clip_parameters()

    def _clip_parameters(self):
        for p in self.parameters():
            p.data.clamp_(-5.0, 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, 1, H, W)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        # collapse spatial dimensions
        features = activations.view(x.size(0), -1)
        # QCNN block
        qcnn_out = self.qcnn(features)
        # optional attention
        if self.use_attention:
            attn_out = self.attention(qcnn_out)
            out = attn_out.squeeze(-1)
        else:
            out = qcnn_out.squeeze(-1)
        # fraud‑style scaling
        out = out * self.scale + self.shift
        if self.clip_weights:
            out = out.clamp(-5.0, 5.0)
        return out

__all__ = ["HybridConv"]
