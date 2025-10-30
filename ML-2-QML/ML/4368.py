"""Hybrid fast estimator combining classical convolution, attention, and sampling modules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """A hybrid classical estimator that integrates a quanvolution filter,
    self‑attention, a sampler, and a linear head.  It can evaluate
    arbitrary observables on the model output and optionally add Gaussian
    shot noise.

    The design mirrors the original FastBaseEstimator but extends it with
    richer feature extraction modules and a flexible noise model.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        noise_shots: Optional[int] = None,
        noise_seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.noise_shots = noise_shots
        self.noise_seed = noise_seed

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] = (),
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables
            Callables that take the model output and return a scalar.
            If ``None`` a single mean observable is used.
        parameter_sets
            A sequence of parameter vectors that will be fed to the model.
        shots
            If provided, Gaussian noise with variance ``1/shots`` is added
            to each observable.  This emulates shot statistics.
        seed
            Random seed for reproducible noise.
        """
        if observables is None:
            observables = [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
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

    @staticmethod
    def default_model() -> nn.Module:
        """Construct a default hybrid model.

        The model consists of:
          * QuanvolutionFilter – 2×2 patch encoding into 4‑dim features.
          * SelfAttention – classical attention over the flattened feature map.
          * SamplerQNN – a small neural sampler that produces a probability
            distribution used as a gating mechanism.
          * Linear head – maps the gated features to 10 classes.
        """
        class QuanvolutionFilter(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                f = self.conv(x)
                return f.view(x.size(0), -1)

        class SelfAttention(nn.Module):
            def __init__(self, embed_dim: int = 4) -> None:
                super().__init__()
                self.embed_dim = embed_dim
                self.q = nn.Linear(embed_dim, embed_dim)
                self.k = nn.Linear(embed_dim, embed_dim)
                self.v = nn.Linear(embed_dim, embed_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                q = self.q(x)
                k = self.k(x)
                v = self.v(x)
                scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
                return scores @ v

        class SamplerQNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 4),
                    nn.Tanh(),
                    nn.Linear(4, 2),
                )

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                return torch.softmax(self.net(inputs), dim=-1)

        # Assemble the hybrid network
        class HybridNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.qfilter = QuanvolutionFilter()
                self.attn = SelfAttention()
                self.sampler = SamplerQNN()
                self.linear = nn.Linear(4 * 14 * 14, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                feat = self.qfilter(x)
                attn_feat = self.attn(feat)
                # Use sampler as a soft gate on the first two features
                gate = self.sampler(feat[:, :2])
                gated = attn_feat * gate
                logits = self.linear(gated)
                return torch.log_softmax(logits, dim=-1)

        return HybridNet()
