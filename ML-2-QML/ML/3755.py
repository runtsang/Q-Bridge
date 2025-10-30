import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    """Dataset generating samples for a smooth periodic regression problem."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = self._generate_data(num_features, samples)

    def _generate_data(self, num_features: int, samples: int):
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {"states": torch.tensor(self.features[idx], dtype=torch.float32),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head that emulates a quantum expectation value."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float):
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        outputs, = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Linear layer followed by HybridFunction to mimic a quantum head."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = x.view(x.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class QuantumRegressionHybrid(nn.Module):
    """Classical regression model with a hybrid quantum‑like head."""
    def __init__(self, num_features: int, hidden_sizes: tuple[int,...] = (32, 16)):
        super().__init__()
        layers = []
        in_size = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.encoder = nn.Sequential(*layers)
        self.hybrid_head = Hybrid(in_features=hidden_sizes[-1] if hidden_sizes else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.hybrid_head(features).squeeze(-1)

__all__ = ["RegressionDataset", "QuantumRegressionHybrid", "Hybrid", "HybridFunction"]
