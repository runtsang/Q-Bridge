"""Hybrid classical classifier integrating feed‑forward, LSTM, self‑attention, and RBF kernel."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, Sequence

# --------------------------------------------------------------------
# 1. Classical classifier backbone (feed‑forward)
# --------------------------------------------------------------------
def build_classifier_circuit(num_features: int, depth: int = 4) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a deep feed‑forward network that mirrors the quantum interface.

    Returns
    -------
    network : nn.Module
        Sequential network comprising linear + ReLU layers.
    encoding : Iterable[int]
        Dummy indices representing classical feature positions.
    weight_sizes : Iterable[int]
        Number of trainable parameters per linear layer.
    observables : list[int]
        Placeholder for output logits (here simply 0 and 1).
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features, bias=True)
        layers.append(linear)
        layers.append(nn.ReLU(inplace=True))
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]  # placeholder for class indices
    return network, encoding, weight_sizes, observables

# --------------------------------------------------------------------
# 2. Classical self‑attention helper
# --------------------------------------------------------------------
class ClassicalSelfAttention:
    """
    Implements a vanilla dot‑product attention block using PyTorch tensors.
    """

    def __init__(self, embed_dim: int, scale: bool = True) -> None:
        self.embed_dim = embed_dim
        self.scale = scale

    def run(self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Compute attention output.

        Parameters
        ----------
        queries, keys, values : np.ndarray
            Input matrices of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Attended representation of shape (batch, seq_len, embed_dim).
        """
        queries_t = torch.as_tensor(queries, dtype=torch.float32)
        keys_t = torch.as_tensor(keys, dtype=torch.float32)
        values_t = torch.as_tensor(values, dtype=torch.float32)

        scores = torch.bmm(queries_t, keys_t.transpose(1, 2))
        if self.scale:
            scores = scores / np.sqrt(self.embed_dim)
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn, values_t)
        return context.detach().numpy()

# --------------------------------------------------------------------
# 3. Classical RBF kernel utilities
# --------------------------------------------------------------------
class KernalAnsatz(nn.Module):
    """
    RBF kernel implemented as a PyTorch module to remain compatible with the quantum interface.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """
    Wrapper that exposes an RBF kernel with a convenient API.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """
    Compute a Gram matrix between two collections of vectors.
    """
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------
# 4. Hybrid classifier model
# --------------------------------------------------------------------
class QuantumHybridClassifier(nn.Module):
    """
    Classical hybrid classifier that unifies a feed‑forward backbone, an LSTM encoder,
    optional self‑attention, and an RBF kernel feature augmentation.
    The interface deliberately mirrors the quantum helper in the seed, enabling
    drop‑in substitution with the quantum variant.
    """

    def __init__(
        self,
        num_features: int,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        use_attention: bool = True,
        kernel_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_features, depth=4)
        self.lstm = nn.LSTM(num_features, lstm_hidden, batch_first=True, num_layers=lstm_layers)
        self.use_attention = use_attention
        if use_attention:
            self.attention = ClassicalSelfAttention(embed_dim=num_features)
        self.kernel = Kernel(gamma=kernel_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, num_features).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, seq_len, 2).
        """
        # LSTM encoder
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        # Optional self‑attention
        if self.use_attention:
            # Convert tensors to numpy for the lightweight attention helper
            queries = lstm_out.detach().numpy()
            keys = lstm_out.detach().numpy()
            values = lstm_out.detach().numpy()
            attn = self.attention.run(queries, keys, values)
            lstm_out = torch.tensor(attn, dtype=torch.float32, device=x.device)
        # Kernel augmentation: compute similarity to a learned prototype set
        batch_size, seq_len, _ = x.shape
        prototypes = torch.randn(batch_size, seq_len, self.num_features, device=x.device)
        kernel_features = torch.stack(
            [self.kernel(x[i, j], prototypes[i, j]) for i in range(batch_size) for j in range(seq_len)]
        ).view(batch_size, seq_len, 1)
        # Concatenate LSTM output with kernel feature
        combined = torch.cat([lstm_out, kernel_features], dim=-1)
        # Feed‑forward classifier
        logits = self.classifier(combined)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuantumHybridClassifier", "build_classifier_circuit", "ClassicalSelfAttention", "Kernel", "kernel_matrix"]
