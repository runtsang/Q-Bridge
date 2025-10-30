from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterable, List, Callable, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Guarantee a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class ConvGen251(nn.Module):
    """
    Classical branch: 2‑D convolution + self‑attention.
    Mirrors the structure of the original Conv.py but
    enriches the feature map with a multi‑head attention
    step, providing richer contextual embeddings.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 embed_dim: int = 4, num_heads: int = 1) -> None:
        super().__init__()
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Multi‑head self‑attention operates on flattened conv output
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (B, 1, H, W)
        Returns:
            tensor of shape (B, embed_dim) – attention‑weighted features
        """
        # Convolution: produce a small feature map
        conv_out = self.conv(x)                         # (B, 1, H', W')
        conv_out = conv_out.view(conv_out.size(0), -1, conv_out.size(-1))
        # Pad or truncate to embed_dim
        seq_len = conv_out.size(1)
        if seq_len < self.embed_dim:
            pad = self.embed_dim - seq_len
            conv_out = F.pad(conv_out, (0, 0, 0, pad))
        elif seq_len > self.embed_dim:
            conv_out = conv_out[:, :self.embed_dim, :]
        # Self‑attention
        attn_out, _ = self.attn(conv_out, conv_out, conv_out)
        # Aggregate over sequence dimension
        return attn_out.mean(dim=1)

    # ------------------------------------------------------------------
    # Evaluation utilities (FastEstimator style)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the model on a batch of parameter sets.
        Each parameter set consists of a single scalar that will
        be used to bias the convolutional weights.
        """
        if observables is None:
            observables = [lambda out: out.mean(dim=-1)]
        if parameter_sets is None:
            parameter_sets = [[0.0]]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                bias = _ensure_batch(params)
                # Temporarily adjust bias
                orig_bias = self.conv.bias.data.clone()
                self.conv.bias.data = bias
                outputs = self.forward(bias)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
                self.conv.bias.data = orig_bias
        return results


__all__ = ["ConvGen251"]
