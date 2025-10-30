"""Hybrid transformer implementation for the classical branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

# ------------------------------------------------------------------
# 1. Utility: autoencoder (re‑used from Autoencoder.py)
# ------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        encoder = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder.append(nn.Linear(in_dim, h))
            encoder.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder.append(nn.Dropout(cfg.dropout))
            in_dim = h
        encoder.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder.append(nn.Linear(in_dim, h))
            decoder.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.decode(self.encode(x))

# ------------------------------------------------------------------
# 2. Utility: fraud‑detection style fully‑connected layer
# ------------------------------------------------------------------
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

def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi], [params.squeeze_r[0], params.squeeze_r[1]]],
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()

class FraudDetectionLayer(nn.Module):
    """A lightweight sequential wrapper around multiple fraud layers."""
    def __init__(self, input_params: FraudLayerParameters, layer_params: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(p, clip=True) for p in layer_params)
        modules.append(nn.Linear(2, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ------------------------------------------------------------------
# 3. Core transformer components (from QTransformerTorch.py)
# ------------------------------------------------------------------
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.pe[:, : x.size(1)]

# ------------------------------------------------------------------
# 4. Unified transformer
# ------------------------------------------------------------------
class HybridTransformerML(nn.Module):
    """Hybrid transformer that can optionally wrap an autoencoder and a fraud‑detection layer."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_autoencoder: bool = False,
        autoencoder_cfg: Optional[AutoencoderConfig] = None,
        use_fraud: bool = False,
        fraud_input: Optional[FraudLayerParameters] = None,
        fraud_layers: Iterable[FraudLayerParameters] = (),
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        self.transformers = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_blocks)]
        )

        self.dropout = nn.Dropout(dropout)

        if num_classes > 2:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, 1)

        # Optional modules
        self.use_autoencoder = use_autoencoder
        if use_autoencoder:
            if autoencoder_cfg is None:
                raise ValueError("autoencoder_cfg must be supplied when use_autoencoder=True")
            self.autoencoder = AutoencoderNet(autoencoder_cfg)
        else:
            self.autoencoder = None

        self.use_fraud = use_fraud
        if use_fraud:
            if fraud_input is None:
                raise ValueError("fraud_input must be supplied when use_fraud=True")
            self.fraud = FraudDetectionLayer(fraud_input, fraud_layers)
        else:
            self.fraud = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token embedding
        x = self.token_embedding(x)
        # Optional fraud module
        if self.use_fraud and self.fraud is not None:
            # Fraud layer expects 2‑dim inputs; we collapse token dim via mean
            x_flat = x.mean(dim=1)  # shape (batch, embed_dim)
            x_f = self.fraud(x_flat)
            # Broadcast back
            x = x_f.unsqueeze(1).repeat(1, x.size(1), 1)
        # Optional autoencoder
        if self.use_autoencoder and self.autoencoder is not None:
            # Encode across token dimension
            x_enc = self.autoencoder.encode(x)
            x = self.autoencoder.decode(x_enc)
        # Positional encoding + transformer
        x = self.pos_embedding(x)
        x = self.transformers(x)
        # Pool
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "AutoencoderNet",
    "AutoencoderConfig",
    "FraudDetectionLayer",
    "FraudLayerParameters",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "HybridTransformerML",
]
