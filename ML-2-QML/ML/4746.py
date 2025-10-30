"""Hybrid Transformer model combining classical transformer blocks, a classical QFC feature extractor,
and a classical classifier head.  The class is fully compatible with the original
QTransformerTorch API but augments it with the following new ideas:

* The `QFCModel` (from QuantumNAT) is used as an optional image encoder.
* The `build_classifier_circuit` function (from QuantumClassifierModel) is
  reimplemented so that the classifier head can be generated automatically
  for an arbitrary number of classes.
* The transformer blocks are now configurable via a simple `mode` flag
  (`classical` or `quantum`) in the quantum variant.  The classical
  version keeps the original, efficient implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Classical transformer primitives – copied and lightly refactored
# ----------------------------------------------------------------------
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding compatible with the original seed."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ----------------------------------------------------------------------
# Optional classical image feature extractor – from QuantumNAT
# ----------------------------------------------------------------------
class QFCModel(nn.Module):
    """Simple CNN + FC that outputs a 4‑dimensional feature vector."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

# ----------------------------------------------------------------------
# Classifier head generator – from QuantumClassifierModel
# ----------------------------------------------------------------------
def build_classifier_circuit_classical(
    embed_dim: int, depth: int, num_classes: int = 2
) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """
    Build a purely classical feed‑forward classifier.

    Parameters
    ----------
    embed_dim : int
        Input feature dimension – should match the transformer output.
    depth : int
        Number of hidden layers.
    num_classes : int
        Number of target classes.

    Returns
    -------
    network : nn.Sequential
        The fully‑connected network.
    encoding : list[int]
        Dummy encoding indices (kept for API parity with the quantum version).
    weight_sizes : list[int]
        Number of trainable parameters per layer.
    observables : list[int]
        Dummy observables – kept for API parity.
    """
    layers: list[nn.Module] = []
    in_dim = embed_dim
    encoding: list[int] = list(range(embed_dim))
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, embed_dim)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = embed_dim

    head = nn.Linear(in_dim, num_classes)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(num_classes))
    return network, encoding, weight_sizes, observables

# ----------------------------------------------------------------------
# HybridTransformer – the unified model
# ----------------------------------------------------------------------
class HybridTransformer(nn.Module):
    """
    A single class that can act as a classical transformer‑based classifier.
    The model optionally embeds image features via `QFCModel` and attaches a
    classifier head generated by `build_classifier_circuit_classical`.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary for text inputs.
    embed_dim : int
        Dimensionality of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer layers.
    ffn_dim : int
        Dimensionality of the hidden layer in the MLP.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Drop‑out probability.
    classifier_depth : int, optional
        Number of hidden layers in the classifier head.
    use_image_features : bool, optional
        If ``True`` the input is expected to be an image tensor and is
        processed by the optional `QFCModel`.  The resulting 4‑dimensional
        vector is projected to ``embed_dim`` before feeding the transformer.
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
        classifier_depth: int = 2,
        use_image_features: bool = False,
        image_feature_dim: int = 4,  # fixed by QFCModel
    ) -> None:
        super().__init__()
        self.use_image_features = use_image_features
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        # Transformer stack
        self.transformers = nn.Sequential(
            *[
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        # Optional image encoder
        if use_image_features:
            self.image_extractor = QFCModel()
            self.image_embedding = nn.Linear(image_feature_dim, embed_dim)

        # Classifier head
        self.classifier_head, _, _, _ = build_classifier_circuit_classical(
            embed_dim, classifier_depth, num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.  ``x`` can be either a batch of token indices
        (``torch.LongTensor``) or a batch of grayscale images
        (``torch.FloatTensor`` of shape ``(B,1,H,W)``).

        Returns
        -------
        logits : torch.Tensor
            Raw logits of shape ``(B, num_classes)``.
        """
        if self.use_image_features:
            # Image → 4‑D feature → embedding → transformer
            feats = self.image_extractor(x)                 # (B,4)
            tokens = self.image_embedding(feats).unsqueeze(1)  # (B,1,embed_dim)
            mask = None
        else:
            tokens = self.token_embedding(x)  # (B,seq_len,embed_dim)
            mask = None

        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))  # pooled representation
        logits = self.classifier_head(x)
        return logits

__all__ = [
    "PositionalEncoder",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "QFCModel",
    "build_classifier_circuit_classical",
    "HybridTransformer",
]
