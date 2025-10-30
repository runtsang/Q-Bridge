import torch
import torch.nn as nn
import numpy as np

class HybridQFCModel(nn.Module):
    """
    Hybrid classical model combining a CNN feature extractor with a flexible
    fully‑connected head.  The head can be configured for classification
    (2 or 4 outputs) or regression (single scalar).  The class also exposes
    helper factories that mirror the quantum counterparts in the QML module.
    """
    def __init__(self, num_classes: int = 4, hidden_dims: tuple[int,...] = (64,)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten_dim = 16 * 7 * 7
        layers = []
        in_dim = self.flatten_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.fc = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        hidden = self.fc(flattened)
        out = self.head(hidden)
        return self.norm(out)

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int):
        """
        Returns a classical feed‑forward classifier and metadata that mimic
        the quantum builder.  The output is a tuple containing the network,
        the indices used for encoding, the weight sizes, and the observables.
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
    def generate_superposition_data(num_features: int, samples: int):
        """
        Generate data in the same style as the quantum regression example:
        random uniform features and a target derived from a trigonometric
        transformation.  Returns a tuple of numpy arrays.
        """
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Simple PyTorch dataset wrapping the superposition data generator.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = HybridQFCModel.generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

__all__ = ["HybridQFCModel", "RegressionDataset"]
