import torch
from torch import nn
import numpy as np

class HybridConvFCL(nn.Module):
    """
    Classical hybrid of a convolutional filter followed by a fully connected layer.
    Designed to be a drop‑in replacement for the quantum quanvolution + fully connected
    pipeline. The module exposes a `run` method that accepts a 2D array and returns a
    scalar output. Parameters are trained via standard PyTorch optimizers.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 conv_threshold: float = 0.0,
                 device: str = "cpu") -> None:
        """
        Args:
            kernel_size: Size of the square filter.
            conv_threshold: Threshold for gating the convolutional output.
            device: Computation device.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.device = device

        # Convolutional part
        self.conv = nn.Conv2d(1, 1, kernel_size, bias=True)

        # Fully connected part
        self.fc = nn.Linear(1, 1)

    def run(self, data: np.ndarray) -> float:
        """
        Execute the hybrid pipeline on a 2D input array.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Scalar output after convolution, gating, and fully connected layers.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        # Add batch and channel dimensions
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        logits = self.conv(tensor)
        gated = torch.sigmoid(logits - self.conv_threshold)
        conv_mean = gated.mean(dim=[2, 3])  # mean over spatial dims
        output = self.fc(conv_mean).tanh()
        return output.item()

__all__ = ["HybridConvFCL"]
