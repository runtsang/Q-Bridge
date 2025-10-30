import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

class QCNNHybrid(nn.Module):
    """
    Classical QCNN hybrid that combines a Quantum‑NAT style feature extractor
    with a stack of fully‑connected layers that emulate the quantum
    convolution and pooling stages.  The design mirrors the classical
    reference but is compact enough to run on a single GPU.
    """
    def __init__(self, nat_output_dim: int = 4, conv_layers: int = 3, seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.nat = self._build_nat()
        self.conv_layers = conv_layers
        self.net = self._build_conv_pool(nat_output_dim, conv_layers)

    def _build_nat(self):
        return nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 4), nn.BatchNorm1d(4)
        )

    def _build_conv_pool(self, in_dim, layers):
        seq = []
        dim = in_dim
        for _ in range(layers):
            seq.append(nn.Linear(dim, dim))
            seq.append(nn.Tanh())
            # simulate pooling by halving the dimensionality
            dim = max(1, dim // 2)
            seq.append(nn.Linear(dim, dim))
            seq.append(nn.Tanh())
        seq.append(nn.Linear(dim, 1))
        return nn.Sequential(*seq)

    def forward(self, x):
        feat = self.nat(x)
        return torch.sigmoid(self.net(feat)).squeeze(-1)


class RegressionDataset(Dataset):
    """
    Dataset of superposition states and corresponding regression targets.
    The data generation follows the pattern used in the quantum reference.
    """
    def __init__(self, samples: int = 1000, features: int = 8):
        self.features, self.labels = generate_superposition_data(features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def generate_superposition_data(num_features: int, samples: int):
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class FastEstimator:
    """
    Lightweight evaluator that runs a model on a list of parameter sets
    and optionally adds Gaussian shot noise to emulate a finite‑shot
    quantum measurement.
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(
        self,
        observables,
        parameter_sets,
        shots: int | None = None,
        seed: int | None = None,
    ):
        if not isinstance(observables, (list, tuple)):
            observables = [observables]
        results = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inp = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                out = self.model(inp)
                row = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(val)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])
        return noisy

__all__ = ["QCNNHybrid", "RegressionDataset", "generate_superposition_data", "FastEstimator"]
