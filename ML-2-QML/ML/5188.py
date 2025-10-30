"""Hybrid transformer classifier with autoencoder, fraud detection, and kernel support (classical)."""

from __future__ import annotations

import math
from typing import Iterable, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
# 1. Autoencoder utilities
# --------------------------------------------------------------------------- #
class AutoencoderConfig:
    """Configuration for the fully‑connected autoencoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """Simple MLP autoencoder with configurable hidden layers."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def Autoencoder(cfg: AutoencoderConfig) -> AutoencoderNet:
    """Factory returning a configured autoencoder."""
    return AutoencoderNet(cfg)


# --------------------------------------------------------------------------- #
# 2. Fraud‑detection sub‑network (classical photonic analogue)
# --------------------------------------------------------------------------- #
class FraudLayerParameters:
    """Parameters for a fraud‑detection layer."""

    def __init__(
        self,
        bs_theta: float,
        bs_phi: float,
        phases: Tuple[float, float],
        squeeze_r: Tuple[float, float],
        squeeze_phi: Tuple[float, float],
        displacement_r: Tuple[float, float],
        displacement_phi: Tuple[float, float],
        kerr: Tuple[float, float],
    ) -> None:
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
            out = out * self.scale + self.shift
            return out

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the photonic fraud‑detection circuit."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# 3. Kernel utilities
# --------------------------------------------------------------------------- #
class KernelAnsatz(nn.Module):
    """Radial‑basis‑function kernel implemented as a neural module."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper that normalises inputs before applying the ansatz."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernelAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor], gamma: float = 1.0) -> torch.Tensor:
    """Compute a Gram matrix using the kernel ansatz."""
    kernel = Kernel(gamma)
    return torch.stack([torch.stack([kernel(x, y) for y in b]) for x in a])


# --------------------------------------------------------------------------- #
# 4. Transformer primitives
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""

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
    """Standard multi‑head attention implemented with PyTorch's built‑in module."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardBase(nn.Module):
    """Base class for the position‑wise feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP with ReLU activation."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block."""

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
    """Sinusoidal positional encoding."""

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


# --------------------------------------------------------------------------- #
# 5. Hybrid text classifier (classical)
# --------------------------------------------------------------------------- #
class HybridTextClassifier(nn.Module):
    """
    Transformer‑based text classifier that can use auto‑encoding, fraud‑detection,
    and a kernel‑based final layer.  The API mirrors the original TextClassifier
    so that the same experiment scripts can be used with either backend.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        *,
        use_autoencoder: bool = False,
        autoencoder_cfg: Optional[AutoencoderConfig] = None,
        use_fraud_detection: bool = False,
        fraud_layers: Optional[Iterable[FraudLayerParameters]] = None,
        use_kernel: bool = False,
        kernel_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        # Optional autoencoder
        if use_autoencoder:
            if autoencoder_cfg is None:
                raise ValueError("autoencoder_cfg must be provided when use_autoencoder=True")
            self.autoencoder = Autoencoder(autoencoder_cfg)
        else:
            self.autoencoder = None

        # Transformer stack
        self.transformers = nn.Sequential(
            *[
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

        # Optional fraud detection branch
        if use_fraud_detection:
            if fraud_layers is None:
                raise ValueError("fraud_layers must be provided when use_fraud_detection=True")
            self.fraud_detection = build_fraud_detection_program(
                FraudLayerParameters(*([0.0] * 8)),  # placeholder input params
                fraud_layers,
            )
        else:
            self.fraud_detection = None

        # Final classifier
        self.dropout = nn.Dropout(dropout)
        if use_kernel:
            self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
            self.kernel = Kernel(kernel_gamma)
        else:
            self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)

        if self.autoencoder is not None:
            x = self.autoencoder.encode(x)

        if self.fraud_detection is not None:
            fraud_score = self.fraud_detection(x.unsqueeze(1))
            x = x + fraud_score.squeeze()

        x = self.dropout(x)

        if hasattr(self, "kernel"):
            # Kernel‑based logits: similarity to class prototypes
            logits = torch.stack([self.kernel(x, proto) for proto in self.prototypes]).t()
            return logits
        else:
            return self.classifier(x)


__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "Autoencoder",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "KernelAnsatz",
    "Kernel",
    "kernel_matrix",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "HybridTextClassifier",
]
