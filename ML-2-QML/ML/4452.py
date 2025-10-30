import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridModel(nn.Module):
    """
    Unified classical model that mimics the structure of the original quantum
    examples while remaining fully differentiable with PyTorch.
    """

    def __init__(self, mode: str = "regression", n_features: int = 1,
                 n_qubits: int = 2, n_wires: int = 4, device: str = "cpu"):
        super().__init__()
        self.mode = mode
        self.device = device

        if mode == "fcl":
            self.layer = nn.Linear(n_features, 1)
        elif mode == "regression":
            self.encoder = nn.Linear(n_wires, n_wires)
            self.random_layer = nn.Linear(n_wires, n_wires)
            self.head = nn.Linear(n_wires, 1)
        elif mode == "classification":
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Dropout2d(p=0.2),
                nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Dropout2d(p=0.5),
                nn.Flatten(),
                nn.Linear(55815, 120),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 1),
            )
            self.hybrid_head = nn.Linear(1, 1)
        elif mode == "quanvolution":
            self.qfilter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
            self.linear = nn.Linear(4 * 14 * 14, 10)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "fcl":
            return torch.tanh(self.layer(x)).mean(dim=0)
        if self.mode == "regression":
            encoded = self.encoder(x)
            random = self.random_layer(encoded)
            out = self.head(random)
            return out.squeeze(-1)
        if self.mode == "classification":
            logits = self.cnn(x)
            probs = torch.sigmoid(self.hybrid_head(logits))
            return torch.cat((probs, 1 - probs), dim=-1)
        if self.mode == "quanvolution":
            features = self.qfilter(x)
            logits = self.linear(features.view(x.size(0), -1))
            return F.log_softmax(logits, dim=-1)

    @staticmethod
    def generate_superposition_data(num_features: int, samples: int):
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    @staticmethod
    def regression_dataset(samples: int, num_features: int):
        x, y = HybridModel.generate_superposition_data(num_features, samples)
        return torch.utils.data.TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
