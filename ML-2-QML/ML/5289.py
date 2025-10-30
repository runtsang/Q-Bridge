"""Hybrid quantum–classical classifier that combines feed‑forward, self‑attention, convolution and sampling."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Classical helpers ----------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """
    Construct a deep feed‑forward classifier and return metadata that mimics the
    quantum helper interface (encoding, weight sizes, observables).
    """
    layers = []
    in_dim = num_features
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, num_features))
        layers.append(nn.ReLU())
        in_dim = num_features
    layers.append(nn.Linear(in_dim, 2))
    network = nn.Sequential(*layers)

    encoding = list(range(num_features))  # placeholder, no real encoding in classical model
    weight_sizes = [l.weight.numel() + l.bias.numel() for l in network if isinstance(l, nn.Linear)]
    observables = [0, 1]  # dummy indices

    return network, encoding, weight_sizes, observables


# --------------------------------------------------------------------------- #
# Self‑attention ------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def SelfAttention():
    class ClassicalSelfAttention:
        """
        A lightweight self‑attention block that mirrors the quantum version.
        The rotation and entangle parameters are treated as learnable tensors.
        """
        def __init__(self, embed_dim: int):
            self.embed_dim = embed_dim
            # parameters are simple tensors; they can be wrapped with nn.Parameter in a larger model
            self.rotation_params = np.random.randn(embed_dim, embed_dim)
            self.entangle_params = np.random.randn(embed_dim, embed_dim)

        def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
            query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            value = torch.as_tensor(inputs, dtype=torch.float32)
            scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
            return (scores @ value).numpy()

    return ClassicalSelfAttention(embed_dim=4)


# --------------------------------------------------------------------------- #
# Convolutional filter ------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def Conv():
    """
    Classic 2‑D convolutional filter that emulates the behaviour of a quanvolution
    layer.  The result is a single scalar that can be concatenated to the feature
    vector before classification.
    """
    class ConvFilter(nn.Module):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(self, data) -> float:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()

    return ConvFilter()


# --------------------------------------------------------------------------- #
# Sampler‑style network ------------------------------------------------------ #
# --------------------------------------------------------------------------- #

def SamplerQNN():
    """
    A tiny neural network that mimics the behaviour of a quantum sampler‑QNN.
    The output is a probability distribution over two classes.
    """
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return F.softmax(self.net(inputs), dim=-1)

    return SamplerModule()


# --------------------------------------------------------------------------- #
# Hybrid model --------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class HybridQuantumClassifier(nn.Module):
    """
    A hybrid classifier that stitches together a classical feed‑forward backbone,
    a self‑attention block, a convolutional filter and a sampler‑style head.
    The interface mirrors the quantum helper so that the same code can be used
    in a quantum‑aware training loop.
    """
    def __init__(self, num_features: int, depth: int) -> None:
        super().__init__()
        self.classifier, self.enc, self.wts, self.obs = build_classifier_circuit(num_features, depth)

        self.attention = SelfAttention()
        self.conv = Conv()
        self.sampler = SamplerQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, num_features)
            Input features.

        Returns
        -------
        logits : torch.Tensor, shape (batch, 2)
            Log‑likelihoods for the two classes.
        """
        # Self‑attention
        att_out = self.attention.run(
            self.attention.rotation_params,
            self.attention.entangle_params,
            x.numpy()
        )
        att_tensor = torch.as_tensor(att_out, dtype=torch.float32)

        # Convolutional filter (expects 2‑D data)
        conv_out = self.conv.run(x.numpy().reshape(-1, 1, 1, 1))
        conv_tensor = torch.as_tensor(conv_out, dtype=torch.float32)

        # Sampler‑style head (uses first two features as a toy example)
        sampler_out = self.sampler(x[:, :2])

        # Concatenate all signals
        combined = torch.cat([x, att_tensor.unsqueeze(1), conv_tensor.unsqueeze(1), sampler_out], dim=1)

        # Feed‑forward classification
        logits = self.classifier(combined)
        return logits


__all__ = ["HybridQuantumClassifier", "build_classifier_circuit", "SelfAttention", "Conv", "SamplerQNN"]
