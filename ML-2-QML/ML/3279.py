import numpy as np
import torch
from torch import nn
from typing import Iterable

class FullyConnectedLayer(nn.Module):
    """Classical fully‑connected layer."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

class ConvFilter(nn.Module):
    """Classical convolutional filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

class HybridLayer:
    """
    Hybrid classical layer that applies a convolutional filter followed by a fully‑connected layer.
    Only the'ml' mode is supported in this module; the quantum counterpart is provided in qml_code.
    """
    def __init__(self, mode: str ='ml', n_features: int = 1,
                 kernel_size: int = 2, threshold: float = 0.0) -> None:
        if mode!='ml':
            raise ValueError("Only'ml' mode is supported in the classical module.")
        self.conv = ConvFilter(kernel_size, threshold)
        self.fc = FullyConnectedLayer(n_features)

    def run(self, data):
        """
        Run the data through the convolution filter and then the fully‑connected layer.
        :param data: 2‑D array of shape (kernel_size, kernel_size) for the convolution step.
        :return: numpy array produced by the fully‑connected layer.
        """
        conv_out = self.conv.run(data)
        # Wrap the single scalar output into an iterable for the FC layer.
        return self.fc.run([conv_out])

__all__ = ["HybridLayer"]
