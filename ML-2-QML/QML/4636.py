import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from dataclasses import dataclass
from typing import Tuple, List, Optional, Iterable

# --------------------------------------------------------------------------- #
#             Quantum quanvolution filter and classifier
# --------------------------------------------------------------------------- #

class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Quantum‑enhanced 2×2 down‑sampling filter using a variational circuit."""
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 4,
                 kernel_size: int = 2,
                 stride: int = 2):
        super().__init__()
        self.n_wires = out_channels * kernel_size * kernel_size
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure  = tq.MeasureAll(tq.PauliZ)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, c, h, w = x.shape
        device = x.device
        patches = []
        for r in range(0, h, self.stride):
            for col in range(0, w, self.stride):
                patch = x[:, :, r:r+self.kernel_size, col:col+self.kernel_size]
                patch = patch.view(bsz, -1)
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, -1))
        return torch.cat(patches, dim=1)

class QuanvolutionClassifierQuantum(nn.Module):
    """Classifier that wraps the quantum filter with a linear head."""
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.filter = QuanvolutionFilterQuantum()
        self.head   = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.filter(x)
        logits = self.head(feat)
        return F.log_softmax(logits, dim=-1)

# --------------------------------------------------------------------------- #
#             Classical auto‑encoder (used in both variants)
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Fully‑connected auto‑encoder (classical)."""
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        enc_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int,
                *,
                latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
#             Quantum‑enhanced transformer primitives
# --------------------------------------------------------------------------- #

class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError('embed_dim must be divisible by num_heads')
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout   = nn.Dropout(dropout)
        self.head_dim  = embed_dim // num_heads

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

class QLayer(tq.QuantumModule):
    """Variational layer used inside the quantum attention block."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params  = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev, x)
        for gate in self.params:
            gate(qdev)
        return self.measure(qdev)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Attention block that injects a variational quantum layer before the classical attention."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 n_wires: int = 8):
        super().__init__(embed_dim, num_heads, dropout)
        self.n_wires = n_wires
        self.q_layer = QLayer(n_wires)
        self.proj    = nn.Linear(n_wires, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, _ = x.shape
        qdev = tq.QuantumDevice(self.n_wires, bsz=s, device=x.device)
        q_out = torch.stack([self.q_layer(x[b, i], qdev) for b in range(b) for i in range(s)], dim=0)
        q_out = q_out.view(b, s, -1)
        x_q  = self.proj(q_out)
        # classical attention on the quantum‑augmented embeddings
        q = self.q_proj(x_q)
        k = self.k_proj(x_q)
        v = self.v_proj(x_q)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, s, self.embed_dim)
        return self.out_proj(out)

class FeedForwardBase(nn.Module):
    """Shared feed‑forward interface."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim   = ffn_dim
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward block that processes each token through a variational circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
            )
            self.params  = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate in self.params:
                gate(qdev)
            return self.measure(qdev)

    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 n_wires: int = 8):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer   = self.QLayer(n_wires)
        self.linear1   = nn.Linear(n_wires, ffn_dim, bias=False)
        self.linear2   = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        qdev = tq.QuantumDevice(self.q_layer.n_wires, bsz=s, device=x.device)
        q_out = torch.stack([self.q_layer(x[b, i], qdev) for b in range(b) for i in range(s)], dim=0)
        q_out = q_out.view(b, s, -1)
        out = self.linear1(self.dropout(q_out))
        return self.linear2(F.relu(out))

class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 n_wires: int = 8):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads,
                                              dropout, n_wires)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout, n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------------- #
#             Quantum‑enhanced hybrid pipeline
# --------------------------------------------------------------------------- #

class HybridQTransformerTorch(nn.Module):
    """Hybrid transformer that can toggle between classical and quantum sub‑modules."""
    def __init__(self,
                 in_features: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum: bool = False,
                 autoencoder_cfg: Optional[AutoencoderConfig] = None,
                 quanv_cfg: Optional[dict] = None):
        super().__init__()
        self.use_quantum = use_quantum

        # Quanvolution – quantum by default
        quan_cfg = quanv_cfg or {}
        self.qfilter = QuanvolutionFilterQuantum(
            in_channels=quan_cfg.get('in_channels', 1),
            out_channels=quan_cfg.get('out_channels', 4),
            kernel_size=quan_cfg.get('kernel_size', 2),
            stride=quan_cfg.get('stride', 2)
        )

        # Auto‑encoder – classical in both variants
        if autoencoder_cfg is None:
            autoencoder_cfg = AutoencoderConfig(in_features)
        self.autoencoder = AutoencoderNet(autoencoder_cfg)

        # Projection to transformer dimension
        self.input_proj = nn.Linear(in_features, embed_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoder(embed_dim)

        # Transformer stack – quantum or classical
        blocks: List[nn.Module] = []
        for _ in range(num_blocks):
            if self.use_quantum:
                blocks.append(TransformerBlockQuantum(embed_dim,
                                                      num_heads,
                                                      ffn_dim,
                                                      dropout,
                                                      n_wires=8))
            else:
                blocks.append(TransformerBlockClassical(embed_dim,
                                                       num_heads,
                                                       ffn_dim,
                                                       dropout))
        self.transformer = nn.Sequential(*blocks)

        # Classifier head
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle image or flat feature input
        if x.dim() == 4:
            feat = self.qfilter(x)
        else:
            feat = x
        feat = self.autoencoder(feat)
        feat = self.input_proj(feat)
        feat = self.pos_enc(feat.unsqueeze(1))
        feat = self.transformer(feat)
        feat = feat.mean(dim=1)
        return self.classifier(feat)

__all__ = [
    'QuanvolutionFilterQuantum',
    'QuanvolutionClassifierQuantum',
    'AutoencoderConfig',
    'AutoencoderNet',
    'Autoencoder',
    'MultiHeadAttentionBase',
    'MultiHeadAttentionQuantum',
    'FeedForwardBase',
    'FeedForwardQuantum',
    'TransformerBlockBase',
    'TransformerBlockQuantum',
    'PositionalEncoder',
    'HybridQTransformerTorch'
]
