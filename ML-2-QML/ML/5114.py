"""Advanced classical estimator combining convolution, self‑attention and a feed‑forward network."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence

# --------------------------------------------------------------------------- #
# 1. Convolutional layer (adapted from Conv.py)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """
    2‑D convolution that mimics a quanvolution filter.
    The output is a single scalar obtained by averaging the sigmoid
    activations over the kernel window.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data is expected to be (batch, 1, H, W)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(1, 2, 3))

# --------------------------------------------------------------------------- #
# 2. Self‑attention helper (adapted from SelfAttention.py)
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """
    Implements a simple scaled dot‑product attention block.
    The module is parameter‑free; the rotation and entangle parameters
    are supplied externally to keep the interface compatible with the
    quantum version.
    """
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        q = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        k = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        v = inputs
        scores = torch.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ v).float()

# --------------------------------------------------------------------------- #
# 3. Fast estimator utilities (adapted from FastBaseEstimator.py)
# --------------------------------------------------------------------------- #
class FastEstimator:
    """
    Evaluates a PyTorch model on a batch of parameter sets.
    Optionally adds Gaussian shot noise to mimic quantum sampling.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]] | None,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        if observables is None:
            observables = [lambda out: out.mean(dim=-1)]

        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().item())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# --------------------------------------------------------------------------- #
# 4. The advanced classical estimator
# --------------------------------------------------------------------------- #
class AdvancedEstimatorQNN(nn.Module):
    """
    A hybrid classical model that stacks:
    1. A 2‑D convolutional filter (ConvFilter)
    2. A self‑attention block (ClassicalSelfAttention)
    3. A small fully‑connected network.
    """

    def __init__(
        self,
        input_dim: int = 2,
        conv_kernel: int = 2,
        attn_dim: int = 4,
        hidden_dims: Sequence[int] = (8, 4),
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel)
        self.attn = ClassicalSelfAttention(embed_dim=attn_dim)

        # Build a tiny MLP that consumes the concatenated conv and attention outputs
        mlp_layers = []
        prev_dim = 1 + attn_dim  # conv output is scalar
        for hd in hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hd))
            mlp_layers.append(nn.Tanh())
            prev_dim = hd
        mlp_layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch, 1, H, W) for convolution; 1D tensor for attention.
        """
        # Convolutional feature
        conv_out = self.conv(x).unsqueeze(-1)  # shape (batch, 1)

        # Prepare dummy parameters for the attention module
        rotation = np.full((self.attn.embed_dim, ), 0.3)
        entangle = np.full((self.attn.embed_dim, ), 0.7)

        # Self‑attention output
        attn_out = self.attn(rotation, entangle, x.reshape(x.shape[0], -1))  # shape (batch, attn_dim)

        # Concatenate features
        features = torch.cat([conv_out, attn_out], dim=-1)

        # MLP regression head
        return self.mlp(features)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]] | None,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Wrapper around FastEstimator to keep the API identical to the quantum side.
        """
        estimator = FastEstimator(self)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = ["AdvancedEstimatorQNN"]
