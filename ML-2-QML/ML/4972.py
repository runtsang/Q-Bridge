import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

@dataclass
class ConvLayerParams:
    weights: torch.Tensor  # shape (out_channels, in_channels, k, k)
    bias: torch.Tensor    # shape (out_channels,)
    threshold: float = 0.0

class QuanvolutionHybrid(nn.Module):
    """
    Classical convolutional filter inspired by the quanvolution example,
    enriched with random initialization, weight clipping, and an optional
    shot‑noise aware evaluator.  The class can be used as a stand‑alone
    classifier or as a building block for deeper hybrid networks.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 2,
        out_channels: int = 4,
        threshold: float = 0.0,
        clip: bool = True,
        clip_bound: float = 5.0,
        classifier_units: int = 10,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.threshold = threshold
        self.clip = clip
        self.clip_bound = clip_bound

        # Convolutional feature extractor
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )
        nn.init.uniform_(self.conv.weight, -1.0, 1.0)
        nn.init.uniform_(self.conv.bias, -1.0, 1.0)
        if self.clip:
            self.conv.weight.data.clamp_(-clip_bound, clip_bound)
            self.conv.bias.data.clamp_(-clip_bound, clip_bound)

        # Linear classifier
        self.linear = nn.Linear(
            out_features=classifier_units,
            in_features=out_channels * ((28 - kernel_size) // stride + 1) ** 2,
        )
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Apply convolution, flatten, and classify."""
        features = self.conv(x)
        features = torch.sigmoid(features - self.threshold)
        flat = features.view(x.size(0), -1)
        logits = self.linear(flat)
        return F.log_softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Evaluation utilities inspired by FastBaseEstimator
    # ------------------------------------------------------------------
    def _ensure_batch(self, values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Evaluate the model over a collection of parameter sets.

        Parameters
        ----------
        observables : iterable of callables
            Functions that map a model output to a scalar (e.g. mean, accuracy).
        parameter_sets : sequence of sequences
            Each inner sequence contains a flattened list of parameters that will
            be loaded into the model's weights and biases before evaluation.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added to each
            observable value to mimic measurement shot noise.
        seed : int, optional
            Random seed for reproducible shot‑noise.

        Returns
        -------
        List of lists containing scalar observable values for each parameter set.
        """
        if not observables:
            observables = [lambda out: out.mean()]

        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                # Load parameters into the model
                self._load_params(params)
                # Dummy input: batch of zeros to trigger forward
                dummy = torch.zeros(1, 1, 28, 28)
                output = self(dummy)
                row: List[float] = []
                for obs in observables:
                    val = float(obs(output).item())
                    row.append(val)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def _load_params(self, flat_params: Sequence[float]) -> None:
        """Convenience for loading flattened parameters into conv and linear."""
        params = torch.as_tensor(flat_params, dtype=torch.float32)
        conv_params = params[: self.conv.numel()]
        lin_params = params[self.conv.numel() : self.conv.numel() + self.linear.numel()]
        # Load conv
        conv_weights = conv_params[: self.conv.weight.numel()]
        conv_bias = conv_params[self.conv.weight.numel() :]
        self.conv.weight.data = conv_weights.reshape(self.conv.weight.shape)
        self.conv.bias.data = conv_bias.reshape(self.conv.bias.shape)
        # Load linear
        lin_weights = lin_params[: self.linear.weight.numel()]
        lin_bias = lin_params[self.linear.weight.numel() :]
        self.linear.weight.data = lin_weights.reshape(self.linear.weight.shape)
        self.linear.bias.data = lin_bias.reshape(self.linear.bias.shape)

__all__ = ["QuanvolutionHybrid"]
