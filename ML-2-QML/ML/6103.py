from __future__ import annotations

import math
from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedQuantumClassifier(nn.Module):
    """
    Classical backbone that can act as a simple feed‑forward network or a
    transformer‑style classifier.  The design is inspired by the
    original `QuantumClassifierModel` and `QTransformerTorch` seeds.
    """
    def __init__(
        self,
        num_features: int,
        depth: int,
        num_classes: int = 2,
        use_transformer: bool = False,
        num_heads: int = 4,
        ffn_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.num_classes = num_classes
        self.use_transformer = use_transformer

        if use_transformer:
            self.embedding = nn.Linear(num_features, ffn_dim)
            self.pos_enc = PositionalEncoder(ffn_dim)
            self.transformer = nn.Sequential(
                *[TransformerBlock(ffn_dim, num_heads, ffn_dim) for _ in range(depth)]
            )
            self.classifier = nn.Linear(ffn_dim, num_classes)
        else:
            layers: List[nn.Module] = []
            in_dim = num_features
            for _ in range(depth):
                layers.append(nn.Linear(in_dim, num_features))
                layers.append(nn.ReLU())
                in_dim = num_features
            layers.append(nn.Linear(in_dim, num_classes))
            self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_transformer:
            x = self.embedding(x)
            x = self.pos_enc(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.classifier(x)
        else:
            return self.backbone(x)

def build_classifier_circuit(
    num_features: int,
    depth: int,
    *,
    num_classes: int = 2,
    use_transformer: bool = False,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Factory that returns a classical classifier matching the original
    signature.  The returned tuple contains the model, a list of
    feature indices, a list of parameter counts per layer, and a list of
    observable indices.
    """
    model = UnifiedQuantumClassifier(
        num_features,
        depth,
        num_classes=num_classes,
        use_transformer=use_transformer,
    )
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in model.parameters()]
    observables = list(range(num_classes))
    return model, encoding, weight_sizes, observables

# Helper transformer components
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

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

__all__ = ["build_classifier_circuit", "UnifiedQuantumClassifier"]
