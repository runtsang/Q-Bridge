"""Hybrid transformer implementation for the quantum branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import math
import torch
from torch import nn
import torchquantum as tq
import torchquantum.functional as tqf

# ------------------------------------------------------------------
# 1. Utility: autoencoder (classical, reused)
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
        return self.decoder(self.encoder(x))

# ------------------------------------------------------------------
# 2. Utility: fraud‑detection style layer (classical)
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
    def __init__(self, input_params: FraudLayerParameters, layer_params: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(p, clip=True) for p in layer_params)
        modules.append(nn.Linear(2, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ------------------------------------------------------------------
# 3. Quantum transformer blocks (from QTransformerTorch.py)
# ------------------------------------------------------------------
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        return x.view(batch, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim // self.num_heads)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, v), scores

    def downstream(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch: int, mask: Optional[torch.Tensor] = None):
        qh = self.separate_heads(q)
        kh = self.separate_heads(k)
        vh = self.separate_heads(v)
        out, w = self.attention(qh, kh, vh, mask)
        self.attn_weights = w
        return out.transpose(1, 2).contiguous().view(batch, -1, self.embed_dim)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for w, g in zip(range(self.n_wires), self.parameters):
                g(q_device, wires=w)
            for w in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[w, w + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = x.size(0)
        if x.size(-1)!= self.embed_dim:
            raise ValueError("Input embed dim mismatch")
        k = self._apply_quantum(x)
        q = self._apply_quantum(x)
        v = self._apply_quantum(x)
        out = self.downstream(q, k, v, b, mask)
        return self.combine(out)

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        # Project into heads and apply quantum circuit per head
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_outs = []
            for h in token.unbind(dim=1):
                qdev = self.q_device.copy(bsz=h.size(0), device=h.device)
                head_outs.append(self.q_layer(h, qdev))
            projections.append(torch.stack(head_outs, dim=1))
        return torch.stack(projections, dim=1)

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

class FeedForwardQuantum(FeedForwardBase):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for w, g in enumerate(self.parameters):
                g(q_device, wires=w)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qheads: int, n_qffn: int, q_device: Optional[tq.QuantumDevice] = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device)
        if n_qffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qffn, dropout)
        else:
            self.ffn = FeedForwardBase(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# ------------------------------------------------------------------
# 4. Unified transformer (quantum flavor)
# ------------------------------------------------------------------
class HybridTransformerQML(nn.Module):
    """Quantum‑enabled transformer that optionally uses an autoencoder and fraud layer."""
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
        n_qheads: int = 0,
        n_qffn: int = 0,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        self.transformers = nn.Sequential(
            *[TransformerBlockQuantum(
                embed_dim, num_heads, ffn_dim,
                n_qheads, n_qffn, q_device, dropout
              ) for _ in range(num_blocks)]
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

        # Optional components
        self.use_autoencoder = use_autoencoder
        if use_autoencoder:
            if autoencoder_cfg is None:
                raise ValueError("autoencoder_cfg required when use_autoencoder=True")
            self.autoencoder = AutoencoderNet(autoencoder_cfg)
        else:
            self.autoencoder = None

        self.use_fraud = use_fraud
        if use_fraud:
            if fraud_input is None:
                raise ValueError("fraud_input required when use_fraud=True")
            self.fraud = FraudDetectionLayer(fraud_input, fraud_layers)
        else:
            self.fraud = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        if self.use_fraud and self.fraud is not None:
            x_flat = x.mean(dim=1)
            x_f = self.fraud(x_flat)
            x = x_f.unsqueeze(1).repeat(1, x.size(1), 1)
        if self.use_autoencoder and self.autoencoder is not None:
            enc = self.autoencoder.encode(x)
            x = self.autoencoder.decode(enc)
        x = self.pos_embedding(x)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "AutoencoderNet",
    "AutoencoderConfig",
    "FraudDetectionLayer",
    "FraudLayerParameters",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "HybridTransformerQML",
]
