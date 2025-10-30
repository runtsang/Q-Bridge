import math
import torch
from torch import nn
from typing import List, Iterable, Tuple

# --------------------------------------------------------------------
# Classical convolutional filter (drop‑in replacement for quanvolution)
# --------------------------------------------------------------------
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

# --------------------------------------------------------------------
# Fraud‑detection style custom layers
# --------------------------------------------------------------------
class FraudLayerParameters:
    def __init__(self,
                 bs_theta: float,
                 bs_phi: float,
                 phases: Tuple[float, float],
                 squeeze_r: Tuple[float, float],
                 squeeze_phi: Tuple[float, float],
                 displacement_r: Tuple[float, float],
                 displacement_phi: Tuple[float, float],
                 kerr: Tuple[float, float]) -> None:
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_head(layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    modules = [_layer_from_params(layer, clip=True) for layer in layers]
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------
# Transformer backbone (classical)
# --------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        q = self.q_linear(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------
# Final estimator (classical regression head)
# --------------------------------------------------------------------
class EstimatorHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------------------------------------------------
# Hybrid model (classical)
# --------------------------------------------------------------------
class HybridModel(nn.Module):
    """Classical hybrid architecture that chains convolution, transformer,
    fraud‑detection head and a regression estimator."""
    def __init__(self,
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0,
                 embed_dim: int = 32,
                 num_heads: int = 4,
                 num_blocks: int = 2,
                 ffn_dim: int = 64,
                 fraud_layers: List[FraudLayerParameters] = []):
        super().__init__()
        self.conv = ConvFilter(conv_kernel, conv_threshold)
        self.proj = nn.Linear(1, embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)]
        )
        self.pos_enc = PositionalEncoder(embed_dim)
        self.fraud_head = build_fraud_head(fraud_layers)
        self.estimator = EstimatorHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W)
        conv_out = self.conv(x)                     # (batch, 1, H', W')
        batch, _, h, w = conv_out.shape
        seq = h * w
        # flatten spatial dims into sequence
        seq_emb = conv_out.view(batch, seq, 1)
        seq_emb = self.proj(seq_emb)                # (batch, seq, embed_dim)
        seq_emb = self.pos_enc(seq_emb)
        seq_emb = self.transformer(seq_emb)          # (batch, seq, embed_dim)
        x = seq_emb.mean(dim=1)                     # (batch, embed_dim)
        x = self.fraud_head(x)                      # (batch, 1)
        x = self.estimator(x)                       # (batch, 1)
        return x

__all__ = ["HybridModel", "FraudLayerParameters"]
