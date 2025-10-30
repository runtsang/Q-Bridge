import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Iterable, List, Optional

# --------------------------------------------------------------------------- #
#             Classical quanvolution filter and classifier
# --------------------------------------------------------------------------- #

class QuanvolutionFilter(nn.Module):
    """2×2 down‑sampling CNN mimicking the quantum filter from the seed."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Classifier that stacks the quanvolution filter with a linear head."""
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.filter = QuanvolutionFilter()
        self.head   = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.filter(x)
        logits = self.head(feat)
        return F.log_softmax(logits, dim=-1)

# --------------------------------------------------------------------------- #
#             Classical auto‑encoder
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Simple fully‑connected auto‑encoder."""
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int, *, latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

def train_autoencoder(model: AutoencoderNet,
                      data: torch.Tensor,
                      *,
                      epochs: int = 100,
                      batch_size: int = 64,
                      lr: float = 1e-3,
                      weight_decay: float = 0.0,
                      device: Optional[torch.device] = None) -> List[float]:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            rec  = model(batch)
            loss = loss_fn(rec, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# --------------------------------------------------------------------------- #
#             Classical transformer primitives
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

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
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

class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

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
#             Hybrid pipeline that can be swapped to a quantum variant
# --------------------------------------------------------------------------- #

class HybridQTransformerTorch(nn.Module):
    """Unified pipeline that fuses quanvolution, auto‑encoding, and transformer layers.
    The constructor accepts a flag to switch to the quantum‑enhanced variant.
    """
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

        # Quanvolution filter
        quan_cfg = quanv_cfg or {}
        self.qfilter = QuanvolutionFilter(
            in_channels=quan_cfg.get('in_channels', 1),
            out_channels=quan_cfg.get('out_channels', 4),
            kernel_size=quan_cfg.get('kernel_size', 2),
            stride=quan_cfg.get('stride', 2)
        )

        # Auto‑encoder
        if autoencoder_cfg is None:
            autoencoder_cfg = AutoencoderConfig(in_features)
        self.autoencoder = AutoencoderNet(autoencoder_cfg)

        # Input projection to transformer dimension
        self.input_proj = nn.Linear(in_features, embed_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoder(embed_dim)

        # Transformer stack
        blocks = []
        for _ in range(num_blocks):
            blocks.append(TransformerBlockClassical(embed_dim, num_heads,
                                                    ffn_dim, dropout))
        self.transformer = nn.Sequential(*blocks)

        # Classifier head
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Image input path (4‑D) or generic feature path (2‑D)
        if x.dim() == 4:
            feat = self.qfilter(x)
        else:
            feat = x
        feat = self.autoencoder(feat)
        feat = self.input_proj(feat)
        feat = self.pos_enc(feat.unsqueeze(1))  # add sequence dimension
        feat = self.transformer(feat)
        feat = feat.mean(dim=1)
        return self.classifier(feat)

__all__ = [
    'QuanvolutionFilter',
    'QuanvolutionClassifier',
    'AutoencoderConfig',
    'AutoencoderNet',
    'Autoencoder',
    'train_autoencoder',
    'MultiHeadAttentionBase',
    'MultiHeadAttentionClassical',
    'FeedForwardBase',
    'FeedForwardClassical',
    'TransformerBlockBase',
    'TransformerBlockClassical',
    'PositionalEncoder',
    'HybridQTransformerTorch'
]
