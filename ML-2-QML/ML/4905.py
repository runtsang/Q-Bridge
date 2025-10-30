import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Tuple, Iterable

# --------------------------------------------------------------------------- #
# Classical self‑attention helper
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    """Simple multi‑head self‑attention using PyTorch tensors."""

    def __init__(self, embed_dim: int, num_heads: int = 1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        scores = torch.einsum("bshd,bsHd->bshH", q, k) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bshH,bsHd->bshd", attn, v)
        out = out.reshape(batch, seq_len, self.embed_dim)
        return self.out_proj(out)

# --------------------------------------------------------------------------- #
# Auto‑encoder
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Fully‑connected auto‑encoder."""

    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def Autoencoder(input_dim: int,
                latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)


# --------------------------------------------------------------------------- #
# Quant convolutional filter (quanvolution)
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution followed by flattening."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Classifier that uses the quanvolution filter."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.filter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.filter(x)
        logits = self.linear(feat)
        return F.log_softmax(logits, dim=-1)


# --------------------------------------------------------------------------- #
# Hybrid pipeline
# --------------------------------------------------------------------------- #
class SelfAttentionGen074:
    """
    Hybrid pipeline that can be run in either classical or quantum mode.

    The classical mode applies a multi‑head self‑attention block, an MLP
    auto‑encoder and a quanvolution classifier.  The quantum mode
    provides a quantum self‑attention circuit, a quantum auto‑encoder
    (SamplerQNN), a quantum fully‑connected model, and a quantum
    quanvolution filter.  The same interface is used for both branches
    to enable side‑by‑side comparison.
    """

    def __init__(self,
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 autoencoder_cfg: AutoencoderConfig | None = None,
                 num_classes: int = 10):
        # Classical sub‑modules
        self.classical_attention = ClassicalSelfAttention(embed_dim, num_heads)
        self.autoencoder = Autoencoder(
            input_dim=embed_dim,
            latent_dim=autoencoder_cfg.latent_dim if autoencoder_cfg else 32,
            hidden_dims=autoencoder_cfg.hidden_dims if autoencoder_cfg else (128, 64),
            dropout=autoencoder_cfg.dropout if autoencoder_cfg else 0.1,
        )
        self.classical_classifier = QuanvolutionClassifier(num_classes=num_classes)

    # ---------- Classical path ------------------------------------------------
    def run_classical(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run data through the classical branch.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, seq_len, embed_dim) for attention
            or (B, 1, 28, 28) for the quanvolution classifier.
        Returns
        -------
        torch.Tensor
            Log‑probabilities from the classifier.
        """
        # If input is 3‑D, treat as sequence for attention
        if x.dim() == 3:
            attn_out = self.classical_attention(x)
            latent = self.autoencoder.encode(attn_out.reshape(x.size(0), -1))
            # reshape back to sequence form
            latent_seq = latent.view(x.size(0), -1, self.autoencoder.encoder[-1].out_features)
            cls_out = self.classical_classifier(latent_seq[:, 0, :].unsqueeze(-1).unsqueeze(-1))
        else:
            # Assume 4‑D image tensor
            cls_out = self.classical_classifier(x)
        return cls_out

    # ---------- Quantum path --------------------------------------------------
    def run_quantum(self, *args, **kwargs):
        """
        Placeholder for quantum execution.  In a real deployment this
        would invoke the quantum sub‑modules defined in the corresponding
        quantum module.  The function is kept to preserve API symmetry.
        """
        raise NotImplementedError("Quantum execution is implemented in the quantum module.")


__all__ = ["SelfAttentionGen074"]
