import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate classical features sampled from a superposition‑inspired distribution."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class ConvFilter(nn.Module):
    """Classical 2‑D convolutional filter that emulates the quantum quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data is expected to be (batch, height, width)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(2, 3), keepdim=True).squeeze(-1)

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid that mimics the quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float):
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Simple dense head that replaces a quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class EstimatorNN(nn.Module):
    """Baseline feed‑forward regressor inspired by EstimatorQNN."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

class HybridRegressionModel(nn.Module):
    """Hybrid classical‑quantum regression model that blends convolution,
    dense layers and a differentiable hybrid head."""
    def __init__(self, num_features: int):
        super().__init__()
        # Convolutional feature extractor (classical analogue of quanvolution)
        self.conv = ConvFilter(kernel_size=2)
        # Dense feature extractor
        self.fc = nn.Sequential(
            nn.Linear(num_features + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Hybrid head
        self.hybrid = Hybrid(1, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (batch, features)
        # reshape for conv: assume features form a square
        size = int(np.sqrt(inputs.shape[1]))
        conv_input = inputs.view(inputs.shape[0], 1, size, size)
        conv_feat = self.conv(conv_input)
        # concatenate conv output with raw features
        x = torch.cat([conv_feat, inputs], dim=1)
        x = self.fc(x)
        return self.hybrid(x)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data", "EstimatorNN"]
