import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------- #
# Positional encoding – identical to the original implementation
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------------- #
# Classical substitutes for quantum sub‑modules
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data):
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

class FullyConnectedLayer(nn.Module):
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas):
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean()
        return expectation

class SamplerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

# --------------------------------------------------------------------------- #
# Classical transformer block
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# Hybrid transformer – API compatible with QTransformerTorch
# --------------------------------------------------------------------------- #
class HybridTransformerML(nn.Module):
    """
    Classical transformer with optional quantum‑inspired sub‑modules.
    The configuration mirrors the original QTransformerTorch API but
    allows toggling of ConvFilter, FullyConnectedLayer and SamplerModule.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 use_conv: bool = False,
                 use_fcl: bool = False,
                 use_sampler: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
             for _ in range(num_blocks)]
        )
        self.use_conv = use_conv
        self.use_fcl = use_fcl
        self.use_sampler = use_sampler
        if use_conv:
            self.conv = ConvFilter()
        if use_fcl:
            self.fcl = FullyConnectedLayer()
        if use_sampler:
            self.sampler = SamplerModule()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        for block in self.blocks:
            x = block(x)
        if self.use_conv:
            conv_outs = []
            for token in x.unbind(dim=1):
                reshaped = token.detach().cpu().numpy().reshape(
                    1, 1, self.conv.kernel_size, self.conv.kernel_size
                )
                conv_val = self.conv(reshaped)
                conv_outs.append(conv_val)
            x = torch.stack(conv_outs, dim=1).to(x.device)
        if self.use_fcl:
            fcl_val = self.fcl(x.mean(dim=1).detach().cpu().numpy())
            x = x + fcl_val.unsqueeze(1)
        if self.use_sampler:
            x = self.sampler(x.mean(dim=1))
            x = x.unsqueeze(1).repeat(1, x.size(1), 1)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = [
    "PositionalEncoder",
    "ConvFilter",
    "FullyConnectedLayer",
    "SamplerModule",
    "TransformerBlock",
    "HybridTransformerML",
]
