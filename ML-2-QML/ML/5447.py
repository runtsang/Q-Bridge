import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

class HybridModel(nn.Module):
    """
    Unified classical neural network that can act as a CNN, classifier, or regressor.
    The architecture is chosen at construction time; metadata (encoding, weight sizes,
    and observables) is exposed for downstream experiments.
    """
    def __init__(self, mode: str = "cnn", num_features: int = 4, depth: int = 2, is_classifier: bool = True):
        super().__init__()
        self.mode = mode
        if mode == "cnn":
            self.net = self.build_cnn()
        elif mode == "classifier":
            self.net, self.encoding, self.weight_sizes, self.observables = self.build_classifier(num_features, depth)
        elif mode == "regressor":
            self.net = self.build_regressor()
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def build_cnn() -> nn.Module:
        """
        Returns a CNN followed by a fully‑connected head.
        """
        features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        return nn.Sequential(features, fc, nn.BatchNorm1d(4))

    @staticmethod
    def build_classifier(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Constructs a feed‑forward classifier mirroring the quantum helper.
        Returns the network, an encoding index list, weight sizes, and output observables.
        """
        layers = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

    @staticmethod
    def build_regressor() -> nn.Module:
        """
        Simple regression network inspired by EstimatorQNN.
        """
        return nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate a list of scalar observables over a batch of parameter sets.
        Optionally inject Gaussian shot noise.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32)
                if inputs.ndim == 1:
                    inputs = inputs.unsqueeze(0)
                outputs = self(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy
        return results
