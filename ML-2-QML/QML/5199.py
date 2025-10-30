import torch
import torch.nn as nn
import torchquantum as tq

# Quantum convolutional frontend
from Quanvolution import QuanvolutionFilter as QuantumConvFilter

# Quantum transformer components
from QTransformerTorch import (
    TransformerBlockQuantum,
    PositionalEncoder,
)

class HybridTransformer(nn.Module):
    """
    Quantum‑enhanced hybrid transformer that processes text or image data.
    It uses a quantum convolutional frontend for images, quantum attention,
    quantum feed‑forward, and an optional quantum auto‑encoder.
    """
    def __init__(self,
                 modality: str = "text",
                 vocab_size: int = 30522,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_blocks: int = 6,
                 ffn_dim: int = 256,
                 num_classes: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        self.modality = modality
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes

        if modality == "text":
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_encoder = PositionalEncoder(embed_dim)
        else:  # image
            self.conv_frontend = QuantumConvFilter()
            self.linear_proj = nn.Linear(4, embed_dim)
            self.pos_encoder = PositionalEncoder(embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlockQuantum(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                n_qubits_transformer=embed_dim,
                n_qubits_ffn=embed_dim,
                n_qlayers=1,
                q_device=None,
                dropout=dropout) for _ in range(num_blocks)
        ])

        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.modality == "text":
            tokens = self.token_embedding(x)
            x = self.pos_encoder(tokens)
        else:
            x = self.conv_frontend(x)
            x = x.view(x.size(0), -1, 4)
            x = self.linear_proj(x)
            x = self.pos_encoder(x)

        for block in self.blocks:
            x = block(x)

        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = ["HybridTransformer"]
