"""Hybrid classical convolutional block (ConvGen86) – reference implementation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# ----------------------------------------------------------------------
# Classical sub‑modules (adapted from the seed files)
# ----------------------------------------------------------------------
def Conv(kernel_size: int = 2, threshold: float = 0.0) -> nn.Module:
    class ConvFilter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(self, data: np.ndarray) -> float:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()

    return ConvFilter()


def SelfAttention(embed_dim: int = 4):
    class ClassicalSelfAttention:
        def __init__(self) -> None:
            self.embed_dim = embed_dim

        def run(
            self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
        ) -> np.ndarray:
            query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            value = torch.as_tensor(inputs, dtype=torch.float32)
            scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
            return (scores @ value).numpy()

    return ClassicalSelfAttention()


def FCL(n_features: int = 1):
    class FullyConnectedLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()


def SamplerQNN():
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


# ----------------------------------------------------------------------
# Hybrid pipeline – the main class
# ----------------------------------------------------------------------
class ConvGen86:
    """
    Classical hybrid convolutional network.

    The pipeline consists of:
    1. Conv → 2D convolution + sigmoid activation.
    2. Self‑attention → attention over a 1‑D embedding of the conv output.
    3. Fully‑connected → learnable linear layer followed by tanh.
    4. SamplerQNN → neural sampler producing a 2‑D probability vector.
    """

    def __init__(self) -> None:
        self.conv = Conv()
        self.attn = SelfAttention()
        self.fc = FCL()
        self.sampler = SamplerQNN()

    def run(
        self,
        data: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        thetas: Iterable[float],
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Execute the full pipeline.

        Parameters
        ----------
        data : np.ndarray
            2‑D array fed to the convolutional filter.
        rotation_params, entangle_params : np.ndarray
            Parameters for the classical self‑attention block.
        thetas : Iterable[float]
            Parameters for the fully‑connected layer.
        inputs : np.ndarray
            2‑D array used by the self‑attention block.

        Returns
        -------
        np.ndarray
            Output of the SamplerQNN (soft‑max probabilities).
        """
        # Convolution step
        conv_out = self.conv.run(data)

        # Self‑attention expects a 2‑D embedding; we tile the conv output
        attn_in = np.tile(conv_out, (1, self.attn.embed_dim))
        attn_out = self.attn.run(rotation_params, entangle_params, attn_in)

        # Fully‑connected step
        fc_out = self.fc.run(thetas)

        # Combine attention and FC outputs into a 2‑D input for the sampler
        sampler_input = torch.tensor(
            [np.concatenate([attn_out.ravel(), fc_out.ravel()])], dtype=torch.float32
        )
        sampler_out = self.sampler(sampler_input)

        return sampler_out.detach().numpy()


__all__ = ["ConvGen86"]
