import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ConvGen622(nn.Module):
    """
    Hybrid classical/quantum convolutional filter with shared regression head.
    The filter can be toggled between a classical Conv2d and a variational quantum circuit.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 use_quantum: bool = False, qbackend: str = "qasm_simulator",
                 shots: int = 100):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum = use_quantum
        if use_quantum:
            # placeholder for quantum filter; actual implementation in qml module
            self.filter = None
        else:
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        # shared regression head
        self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        if self.use_quantum:
            # This branch is only available when the quantum backend is provided
            raise NotImplementedError("Quantum forward requires qml module.")
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        # flatten to scalar
        feat = activations.view(activations.size(0), -1).mean(dim=1, keepdim=True)
        return self.head(feat).squeeze(-1)

    def run(self, data: np.ndarray) -> float:
        """Run the classical filter on a single 2D array."""
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return self.forward(tensor).item()

def Conv(kernel_size: int = 2, threshold: float = 0.0) -> ConvGen622:
    """Backward‑compatible factory returning a classical ConvGen622 instance."""
    return ConvGen622(kernel_size=kernel_size, threshold=threshold, use_quantum=False)

__all__ = ["ConvGen622", "RegressionDataset", "generate_superposition_data", "Conv"]
