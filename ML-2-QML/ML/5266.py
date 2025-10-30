import torch
import torch.nn as nn
import numpy as np

class ConvFilter(nn.Module):
    """Classical emulation of a quantum convolution filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data):
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

class FCLayer(nn.Module):
    """Classical surrogate for a quantum fully‑connected layer."""
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas):
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

class SharedClassName(nn.Module):
    """Hybrid classical neural network that fuses convolution, quantum‑inspired fully‑connected, and optional quantum filter."""
    def __init__(self, use_quantum_filter: bool = False, use_quantum_fcl: bool = False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fclayer = FCLayer()
        self.conv_filter = ConvFilter() if use_quantum_filter else None
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)
        self.use_quantum_fcl = use_quantum_fcl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        if self.conv_filter is not None:
            # Simplified application of the quantum filter to the flattened features
            filt_out = torch.tensor([self.conv_filter.run(f.cpu().numpy()) for f in features.view(bsz, -1)], device=x.device)
            features = torch.cat([features.view(bsz, -1), filt_out.unsqueeze(1)], dim=1)
        flattened = features.view(bsz, -1)
        if self.use_quantum_fcl:
            linear_out = self.fc[0](flattened)
            out = torch.tanh(linear_out).mean(dim=1, keepdim=True)
        else:
            out = self.fc(flattened)
        return self.norm(out)

__all__ = ["SharedClassName"]
