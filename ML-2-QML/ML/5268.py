from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Sequence, Callable, List

# Classical fully‑connected layer mimicking the quantum FCL
def FCL(n_features: int = 1) -> nn.Module:
    """Return a module with a ``run`` method that computes a noisy expectation."""
    class FullyConnectedLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().cpu().numpy()
    return FullyConnectedLayer()

# Dataset utilities
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# Fast estimator for classical models
class FastEstimator:
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(
        self,
        observables: Sequence[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        results: List[List[float]] = []
        self.model.eval()
        rng = np.random.default_rng(seed)
        for params in parameter_sets:
            inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
            outputs = self.model(inputs)
            row: List[float] = []
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.mean().item()
                row.append(float(val))
            if shots is not None:
                row = [float(rng.normal(v, max(1e-6, 1 / shots))) for v in row]
            results.append(row)
        return results

# Core hybrid model
class QuanvolutionGen152(nn.Module):
    """
    Hybrid classical model inspired by the original quanvolution.
    Uses a 2×2 patch convolution followed by a random linear map and a fully connected head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        num_classes: int = 10,
    ):
        super().__init__()
        # 2×2 convolution to extract patches
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        # Random feature map to emulate quantum kernel
        self.patch_map = nn.Linear(out_channels * kernel_size * kernel_size, out_channels, bias=False)
        with torch.no_grad():
            self.patch_map.weight.copy_(torch.randn_like(self.patch_map.weight))
        self.activation = nn.Tanh()
        # Fully connected head
        self.head = nn.Linear(out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract patches
        patches = self.conv(x)  # shape [B, C, H, W]
        B, C, H, W = patches.shape
        # Flatten patches to vectors
        patches = patches.view(B, C * H * W)
        # Apply random map
        features = self.patch_map(patches)
        features = self.activation(features)
        # Flatten features
        features = features.view(B, -1)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)

    def evaluate(
        self,
        observables: Sequence[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Convenience wrapper that mimics FastEstimator.evaluate."""
        return FastEstimator(self).evaluate(observables, parameter_sets)
