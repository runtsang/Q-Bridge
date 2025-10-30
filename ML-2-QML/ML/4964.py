"""Hybrid quanvolution module for classical + quantum processing.

The implementation fuses:
* a lightweight 2×2 convolutional filter (classical),
* a quantum kernel that maps each patch to a 4‑dimensional feature vector
  using a parameterised 4‑qubit circuit (torchquantum),
* a self‑attention block that aggregates the patch features,
* a FastEstimator style noise wrapper for Monte‑Carlo evaluation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Optional

# Import the quantum filter from the QML side.
# The relative import assumes that the QML module resides in the same package.
from.Quanvolution__gen193_qml import QuanvolutionQuantumFilter

@dataclass
class FraudLayerParameters:
    """Parameter set used to build a single quantum layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class SelfAttentionBlock(nn.Module):
    """Classical self‑attention that operates on a sequence of patch embeddings."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        scores = F.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

class QuanvolutionFilter(nn.Module):
    """2×2 convolution that produces a 4‑channel feature map."""

    def __init__(self, in_channels: int = 1, out_channels: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class FastEstimator:
    """
    Evaluate a PyTorch model on a batch of parameter sets.
    Optional Gaussian shot noise can be injected to mimic quantum hardware.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[callable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        self.model.eval()
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [rng.normal(mean, max(1e-6, 1 / shots)) for mean in row]
                noisy.append(noisy_row)
            return noisy
        return results

class QuanvolutionHybrid(nn.Module):
    """
    Classic‑quantum hybrid classifier that combines:
    1. 2×2 convolutional extraction,
    2. Quantum kernel applied patch‑wise,
    3. Optional self‑attention aggregation,
    4. Linear head producing 10‑class logits.
    """

    def __init__(self, *, use_attention: bool = True, n_qubits: int = 4, n_layers: int = 1) -> None:
        super().__init__()
        self.classical = QuanvolutionFilter()
        self.quantum = QuanvolutionQuantumFilter(n_wires=n_qubits, n_layers=n_layers)
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttentionBlock(embed_dim=4)
        else:
            self.attention = None
        self.linear = nn.Linear(4 * 14 * 14 * 2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical pathway
        cls_features = self.classical(x)          # shape: (B, 4*14*14)
        # Quantum pathway
        q_features = self.quantum(x)              # shape: (B, 4*14*14)
        # Concatenate
        feats = torch.cat([cls_features, q_features], dim=1)  # (B, 8*14*14)
        if self.use_attention:
            # Reshape to (B, seq_len, embed_dim)
            seq_len = feats.shape[1] // 4
            embed_dim = 4
            feats = feats.view(-1, seq_len, embed_dim)
            # Simple self‑attention over the sequence
            feats = self.attention(feats, feats, feats)
            feats = feats.view(-1, seq_len * embed_dim)
        logits = self.linear(feats)
        return F.log_softmax(logits, dim=-1)

__all__ = [
    "FraudLayerParameters",
    "SelfAttentionBlock",
    "QuanvolutionFilter",
    "FastEstimator",
    "QuanvolutionHybrid",
]
