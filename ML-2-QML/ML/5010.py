import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, List, Sequence, Tuple

class UnifiedClassifier(nn.Module):
    """Classical feed‑forward classifier mirroring the quantum helper interface."""
    def __init__(self, num_features: int, depth: int = 3) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)
        # Store weight sizes for debugging / introspection
        self.weight_sizes = [p.numel() for p in self.parameters()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the model on a list of input vectors.
        Supports optional shot‑noise simulation matching the FastEstimator design.
        """
        inputs = torch.tensor(parameter_sets, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.net(inputs)
        if shots is None:
            return outputs.cpu().numpy().tolist()
        rng = np.random.default_rng(seed)
        noisy = outputs.cpu().numpy() + rng.normal(0, 1 / shots, outputs.shape)
        return noisy.tolist()

    @staticmethod
    def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthetic dataset used in the regression example.
        Mirrors the data generation logic from the regression reference pair.
        """
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)
