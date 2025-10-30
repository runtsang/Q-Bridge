import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence, Optional

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudQuantumTransformer",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
]

# --------------------------------------------------------------------------- #
# 1.  Classical fraud‑detection MLP (expanded from the original seed)
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Container for a single photonic layer, reused for both classical and quantum parts."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to ±bound, used for weight‑bias and quantum‑parameter clipping."""
    return max(-bound, min(bound, value))


def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool = False,
    activation: nn.Module = nn.Tanh(),
) -> nn.Module:
    """Build a single linear‑plus‑activation‑scale‑shift layer."""
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)

    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Return a PyTorch Sequential that mirrors the photonic circuit."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules += [_layer_from_params(l, clip=True) for l in layers]
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# 2.  Quantum‑enhanced transformer block (placeholder implementations)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(nn.Module):
    """Placeholder for a quantum‑encoded attention module."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        # In a real quantum implementation this would be replaced with a
        # variational circuit that encodes each head.
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        # Simplified scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return self.combine(out)


class FeedForwardQuantum(nn.Module):
    """Placeholder for a quantum‑based feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockQuantum(nn.Module):
    """Transformer block that can use either classical or quantum sub‑modules."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        if use_quantum:
            self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout)
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(self.attn, nn.MultiheadAttention):
            attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        else:
            attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class FraudQuantumTransformer(nn.Module):
    """Hybrid model combining the photonic fraud‑detection MLP with a transformer."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_blocks: int = 2,
        use_quantum: bool = False,
    ):
        super().__init__()
        self.fraud_mlp = build_fraud_detection_program(input_params, layers)
        self.transformer = nn.Sequential(
            *[
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout=0.1,
                    use_quantum=use_quantum,
                )
                for _ in range(num_blocks)
            ]
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be a sequence of 2‑dim vectors (e.g., transaction features)
        mlp_out = self.fraud_mlp(x)
        # For the transformer we need a 3‑D tensor: (batch, seq_len, embed_dim)
        # Here we simply broadcast mlp_out to a sequence length of 1.
        seq = mlp_out.unsqueeze(1)
        transformer_out = self.transformer(seq)
        pooled = transformer_out.mean(dim=1)
        return self.classifier(pooled)
