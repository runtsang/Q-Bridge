import torch
import torch.nn as nn
import torch.nn.functional as F

# Classical and quantum convolutional front‑ends
from Conv import Conv as ClassicalConv
from Quanvolution import QuanvolutionFilter as QuantumConvFilter

# Transformer components
from QTransformerTorch import (
    MultiHeadAttentionClassical,
    MultiHeadAttentionQuantum,
    FeedForwardClassical,
    FeedForwardQuantum,
    PositionalEncoder,
    TransformerBlockClassical,
    TransformerBlockQuantum,
)

# Auto‑encoder utilities
from Autoencoder import Autoencoder, AutoencoderConfig

class HybridTransformer(nn.Module):
    """
    A hybrid transformer that can process text or image data.
    It optionally uses quantum sub‑modules for attention and feed‑forward,
    and includes a quantum or classical convolutional frontend.
    An optional auto‑encoder can be attached for pre‑training or dimensionality reduction.
    """
    def __init__(self,
                 modality: str = "text",
                 vocab_size: int = 30522,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_blocks: int = 6,
                 ffn_dim: int = 256,
                 num_classes: int = 10,
                 dropout: float = 0.1,
                 use_quantum_attention: bool = False,
                 use_quantum_ffn: bool = False,
                 use_quantum_conv: bool = False,
                 use_autoencoder: bool = False,
                 autoencoder_cfg: AutoencoderConfig | None = None):
        super().__init__()
        self.modality = modality
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes

        # Front‑end
        if modality == "text":
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_encoder = PositionalEncoder(embed_dim)
        elif modality == "image":
            if use_quantum_conv:
                self.conv_frontend = QuantumConvFilter()
            else:
                self.conv_frontend = nn.Conv2d(1, 4, kernel_size=2, stride=2)
            self.linear_proj = nn.Linear(4, embed_dim)
            self.pos_encoder = PositionalEncoder(embed_dim)
        else:
            raise ValueError("modality must be 'text' or 'image'")

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if use_quantum_attention or use_quantum_ffn:
                block = TransformerBlockQuantum(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    n_qubits_transformer=embed_dim,
                    n_qubits_ffn=embed_dim if use_quantum_ffn else 0,
                    n_qlayers=1,
                    q_device=None,
                    dropout=dropout)
            else:
                block = TransformerBlockClassical(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout)
            self.blocks.append(block)

        # Optional auto‑encoder
        if use_autoencoder:
            cfg = autoencoder_cfg or AutoencoderConfig(
                input_dim=embed_dim,
                latent_dim=embed_dim // 2,
                hidden_dims=(ffn_dim, embed_dim // 2),
                dropout=dropout)
            self.autoencoder = Autoencoder(cfg)
        else:
            self.autoencoder = None

        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor. For text: (batch, seq_len). For image: (batch, 1, 28, 28).
        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        """
        if self.modality == "text":
            tokens = self.token_embedding(x)          # (B, T, E)
            x = self.pos_encoder(tokens)
        else:  # image
            x = self.conv_frontend(x)                 # (B, 4, 14, 14)
            x = x.view(x.size(0), -1, 4)              # (B, 196, 4)
            x = self.linear_proj(x)                   # (B, 196, E)
            x = self.pos_encoder(x)

        for block in self.blocks:
            x = block(x)

        if self.autoencoder:
            x = self.autoencoder.encode(x)

        x = x.mean(dim=1)          # (B, E)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = ["HybridTransformer"]
