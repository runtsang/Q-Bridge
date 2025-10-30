from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import math

# --- Classical transformer primitives ------------------------------------
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError(f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})")
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)

        def separate_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        k = separate_heads(k)
        q = separate_heads(q)
        v = separate_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(out)

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
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

# --- LSTM helper ---------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# --- Network builder ------------------------------------------------------
def build_classifier_circuit(
    num_features: int,
    depth: int,
    architecture: str = "mlp",
    hidden_dim: int = 32,
    num_heads: int = 4,
    num_blocks: int = 2,
    ffn_dim: int = 64,
    lstm_hidden_dim: int = 64,
    dropout: float = 0.1,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a classical classifier network and return metadata for
    compatibility with the quantum counterpart.
    """
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    if architecture == "mlp":
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, hidden_dim)
            layers.append(linear)
            layers.append(nn.ReLU())
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = hidden_dim
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)

    elif architecture == "transformer":
        layers: List[nn.Module] = [nn.Linear(num_features, hidden_dim)]
        weight_sizes.append(layers[0].weight.numel() + layers[0].bias.numel())
        for _ in range(num_blocks):
            block = TransformerBlockClassical(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            layers.append(block)
            weight_sizes.append(
                sum(p.numel() for p in block.parameters() if p.requires_grad)
            )
        layers.append(nn.Linear(hidden_dim, 2))
        weight_sizes.append(layers[-1].weight.numel() + layers[-1].bias.numel())
        network = nn.Sequential(*layers)

    elif architecture == "lstm":
        lstm_cls = LSTMClassifier(num_features, lstm_hidden_dim)
        weight_sizes.append(
            sum(p.numel() for p in lstm_cls.parameters() if p.requires_grad)
        )
        network = lstm_cls

    else:
        raise ValueError(f"Unknown architecture {architecture!r}")

    observables = list(range(2))
    return network, encoding, weight_sizes, observables

# --- Core model ------------------------------------------------------------
class QuantumClassifierModel:
    """
    Classical implementation of a neural classifier with interchangeable
    back‑ends.  The API mirrors the quantum version so that the same class
    name can be imported from either the classical or quantum module.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        architecture: str = "mlp",
        hidden_dim: int = 32,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 64,
        lstm_hidden_dim: int = 64,
        dropout: float = 0.1,
        use_quantum: bool = False,
    ) -> None:
        self.architecture = architecture
        self.network, _, _, _ = build_classifier_circuit(
            num_features=input_dim,
            depth=hidden_dim,
            architecture=architecture,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ffn_dim=ffn_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            dropout=dropout,
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture == "lstm" and x.ndim == 2:
            x = x.unsqueeze(1)
        return self.network(x)

# --- Dataset helpers ------------------------------------------------------
class ClassificationDataset(Dataset):
    """
    Simple toy classification dataset that mirrors the structure used in
    the quantum regression example but with binary labels.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_classification_data(num_features, samples)

    def __len__(self) -> int:  # pragma: no cover
        return len(self.features)

    def __getitem__(self, index: int):  # pragma: no cover
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.long),
        }

def generate_classification_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random feature vectors and binary labels.
    The label is 1 if the sum of the features exceeds a random threshold
    otherwise 0.  This introduces a non‑linear decision boundary.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    threshold = np.random.uniform(-0.5, 0.5)
    y = (x.sum(axis=1) > threshold).astype(np.int64)
    return x, y

__all__ = [
    "QuantumClassifierModel",
    "ClassificationDataset",
    "generate_classification_data",
]
