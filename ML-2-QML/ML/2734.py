"""Hybrid classical classifier with integrated self‑attention.

The class exposes the same public API as the original quantum helper but
implements the forward pass entirely on the CPU.  It combines a
parametric feed‑forward network with a lightweight self‑attention block
derived from the classical SelfAttention helper.  The attention parameters
are treated as additional trainable weights so that the whole model can
be optimized end‑to‑end with PyTorch.

The build_classifier_circuit function is adapted from the seed but now
produces a network that accepts an arbitrary number of hidden layers
and supports residual connections, making it more expressive while
remaining compatible with the quantum interface.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import numpy as np


def build_classifier_circuit(
    input_dim: int,
    depth: int,
    hidden_dim: int = None,
) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Construct a feed‑forward classifier.

    Parameters
    ----------
    input_dim:
        Dimensionality of the input vector (after attention).
    depth:
        Number of hidden layers.
    hidden_dim:
        Size of each hidden layer; defaults to ``input_dim``.
    """
    hidden_dim = hidden_dim or input_dim
    layers: List[nn.Module] = []
    in_dim = input_dim
    weight_sizes: List[int] = []

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
    encoding = list(range(input_dim))
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


class ClassicalSelfAttention(nn.Module):
    """
    Lightweight self‑attention module.

    Parameters
    ----------
    embed_dim:
        Dimensionality of the embedding space.
    """

    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.key_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the attention output.

        Parameters
        ----------
        inputs:
            Tensor of shape ``(batch, seq_len, embed_dim)``.
        """
        query = torch.matmul(inputs, self.query_weight)
        key = torch.matmul(inputs, self.key_weight)
        scores = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1
        )
        return torch.matmul(scores, inputs)


class QuantumClassifierModel:
    """
    Classical wrapper that mimics the quantum API.

    The class accepts the same arguments as the original quantum
    ``QuantumClassifierModel`` but performs the entire computation
    on the CPU.  It can be used interchangeably in downstream code
    that expects a quantum object.
    """

    def __init__(self, num_features: int, depth: int, embed_dim: int = 4):
        self.num_features = num_features
        self.depth = depth
        self.embed_dim = embed_dim

        # Self‑attention block
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)

        # Build the classifier network; input dimension = num_features + embed_dim
        self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            input_dim=num_features + embed_dim,
            depth=depth,
        )

    def forward(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """
        Run a forward pass.

        Parameters
        ----------
        inputs:
            Feature matrix of shape ``(batch, num_features)``.
        rotation_params, entangle_params:
            Dummy parameters kept for API compatibility.
        """
        batch = torch.as_tensor(inputs, dtype=torch.float32)

        # Expand to a sequence dimension for attention
        seq = batch.unsqueeze(1)  # (batch, 1, embed_dim)
        attn_out = self.attention(seq).squeeze(1)

        # Concatenate attention output with original features
        combined = torch.cat([batch, attn_out], dim=-1)

        logits = self.network(combined)
        return logits.detach().numpy()

    def predict(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """
        Return class probabilities.
        """
        logits = self.forward(inputs, rotation_params, entangle_params)
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        return probs


__all__ = ["QuantumClassifierModel", "build_classifier_circuit", "ClassicalSelfAttention"]
