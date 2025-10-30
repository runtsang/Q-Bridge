import torch
from torch import nn
import numpy as np

class ConvFilter(nn.Module):
    """Emulates the quantum filter with classical PyTorch operations."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply convolution to a single image and return mean activation."""
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

    def run(self, data: np.ndarray) -> float:
        """Convenience wrapper that accepts a NumPy array."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        return self.forward(tensor).item()

class QCNNModel(nn.Module):
    """Stack of fully connected layers mirroring the QCNN helper."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridConvQCNN(nn.Module):
    """Hybrid classical model that chains the Conv filter with a QCNN‑style network."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.conv_filter = ConvFilter(kernel_size, threshold)
        self.qcnn = QCNNModel()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Batch of images with shape (B, 1, H, W).  Currently only a 2×2 patch is
            processed by the Conv filter; the resulting scalar is replicated to
            match the 8‑dimensional input of the QCNN.
        Returns
        -------
        torch.Tensor
            Output of the QCNN head with shape (B, 1).
        """
        batch_size = data.shape[0]
        outs = []
        for i in range(batch_size):
            img = data[i].cpu().numpy().squeeze()
            conv_val = self.conv_filter.run(img)
            x = torch.full((1, 8), conv_val, dtype=torch.float32)
            outs.append(self.qcnn(x))
        return torch.cat(outs, dim=0)

    def run(self, data: np.ndarray) -> float:
        """
        Convenience method to run a single 2×2 image without a PyTorch tensor.
        """
        with torch.no_grad():
            img = torch.as_tensor(data, dtype=torch.float32)
            return self.forward(img.unsqueeze(0)).item()

__all__ = ["ConvFilter", "QCNNModel", "HybridConvQCNN"]
