import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional

# --------------------------------------------------------------------------- #
# Classical transformer backbone (borrowed from QTransformerTorch)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented classically."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        # split heads
        k = k.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        q = q.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.combine_heads(out)


class FeedForwardClassical(nn.Module):
    """Two‑layer MLP feed‑forward."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1, use_bias: bool = False):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=use_bias)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """A single transformer block with optional residuals."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1, use_bias: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout, use_bias)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout, use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
# Photonic Fraud‑Detection sub‑module (borrowed from FraudDetection)
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single photonic layer as a classical linear‑activation‑scale block."""
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
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters]
) -> nn.Sequential:
    """Build a sequential photonic‑style network."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules += [_layer_from_params(layer, clip=True) for layer in layers]
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# Hybrid fusion layer
# --------------------------------------------------------------------------- #
class FraudTransformerFusion(nn.Module):
    """
    A hybrid model that runs a transformer on raw transaction features and a
    photonic‑style fraud‑detection network in parallel,
    then fuses their representations via a learnable weighted sum.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        fraud_params: Iterable[FraudLayerParameters],
        num_classes: int = 2,
        dropout: float = 0.1,
        use_bias: bool = False
    ):
        super().__init__()
        # Transformer backbone
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout, use_bias)
              for _ in range(num_blocks)]
        )
        # Photonic sub‑network
        self.fraud_net = build_fraud_detection_program(
            fraud_params[0], fraud_params[1:]
        )
        # Final classifier
        self.classifier = nn.Linear(embed_dim + 1, num_classes if num_classes > 2 else 1)
        # Learnable fusion weight
        self.fusion_weight = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch, seq_len, embed_dim)
        Returns logits or probabilities depending on num_classes.
        """
        # Transformer path
        t_out = self.transformer(x)          # (B, L, E)
        t_out = t_out.mean(dim=1)            # pooling
        # Fraud‑sensing path
        # flatten to (B*L, 2) and reshape to (B, L, 2)
        seq_len = x.shape[1]
        flat = x.reshape(-1, 2)  # assume last two dims encode two‑feature pair
        fraud_out = self.fraud_net(flat)  # (B*L, 1)
        fraud_out = fraud_out.reshape(-1, seq_len, 1)  # (B, L, 1)
        # Fuse
        fused = self.fusion_weight * t_out + (1 - self.fusion_weight) * fraud_out.squeeze(-1)
        # Final classification
        logits = self.classifier(torch.cat([fused, t_out], dim=-1))
        return logits
