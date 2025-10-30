"""Hybrid binary classifier with classical backbone and classical head.

This module builds on the original ClassicalQuantumBinaryClassification example,
adding a transformer encoder and a fully‑connected quantum‑style layer
implemented classically.  It is fully compatible with the original
anchor but exposes a richer feature extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Tuple

# ----------------------------------------------------------------------
# Data utilities – adapted from QuantumRegression.py
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for binary classification.

    The labels are obtained by thresholding a sinusoidal function of the
    input angles, mirroring the regression example but converted to a
    binary task.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    # Convert to binary labels
    y = (y > 0).astype(np.float32)
    return x, y

class ClassificationDataset(torch.utils.data.Dataset):
    """Dataset wrapper for the synthetic binary data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# ----------------------------------------------------------------------
# Classical hybrid layers – adapted from FCL.py
# ----------------------------------------------------------------------
class FCL(nn.Module):
    """Classical stand‑in for a fully‑connected quantum layer."""
    def __init__(self, in_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

# ----------------------------------------------------------------------
# Hybrid activation – adapted from HybridFunction in the seed
# ----------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics a quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None

class Hybrid(nn.Module):
    """Classical head that replaces the quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

# ----------------------------------------------------------------------
# CNN backbone – unchanged from the seed
# ----------------------------------------------------------------------
class CNNBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ----------------------------------------------------------------------
# Transformer encoder – classical version from QTransformerTorch.py
# ----------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Simple transformer block with multi‑head attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerEncoder(nn.Module):
    """Stack of transformer blocks."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, num_blocks: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

# ----------------------------------------------------------------------
# Full hybrid classifier
# ----------------------------------------------------------------------
class HybridBinaryClassifier(nn.Module):
    """CNN + transformer + classical hybrid head for binary classification."""
    def __init__(self,
                 num_heads: int = 4,
                 ffn_dim: int = 128,
                 num_blocks: int = 2,
                 shift: float = 0.0):
        super().__init__()
        self.backbone = CNNBackbone()
        self.transformer = TransformerEncoder(1, num_heads, ffn_dim, num_blocks)
        self.hybrid = Hybrid(1, shift=shift)
        # Final projection to two logits
        self.proj = nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        x = self.backbone(x)
        # Reshape to sequence for transformer (batch, seq_len=1, embed_dim=1)
        seq = x.unsqueeze(1)
        seq = self.transformer(seq)
        # Hybrid head
        logits = self.hybrid(seq.squeeze(1))
        probs = torch.cat([logits, 1 - logits], dim=-1)
        return probs

__all__ = [
    "HybridBinaryClassifier",
    "ClassificationDataset",
    "generate_superposition_data",
    "HybridFunction",
    "Hybrid",
    "FCL",
]
