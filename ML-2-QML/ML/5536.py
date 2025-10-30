import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvFilter(nn.Module):
    """Classical 2‑D convolutional filter emulating a quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Expect data shape (batch, 1, H, W)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2,3])  # mean over spatial dims

class SamplerQNN(nn.Module):
    """Hybrid classical sampler that mirrors the quantum interface."""
    def __init__(self,
                 num_features: int,
                 num_qubits: int,
                 depth: int,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.conv = ConvFilter(kernel_size, threshold).to(device)
        self.encoder = nn.Linear(num_features, num_qubits).to(device)
        self.classifier = nn.Linear(num_qubits, 2).to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, 1, H, W) or (batch, num_features).

        Returns
        -------
        torch.Tensor
            Probabilities over two classes, shape (batch, 2).
        """
        if inputs.dim() == 4:  # image‑like
            conv_out = self.conv(inputs).unsqueeze(-1)  # (batch, 1)
            features = conv_out.repeat(1, self.encoder.in_features)
        else:
            features = inputs
        encoded = self.encoder(features)
        logits = self.classifier(encoded)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> np.ndarray:
        """
        Draw classical samples from the output distribution.

        Parameters
        ----------
        inputs : torch.Tensor
            Input batch.
        num_samples : int
            Number of samples per batch element.

        Returns
        -------
        np.ndarray
            Sampled labels of shape (batch, num_samples).
        """
        probs = self.forward(inputs).detach().cpu().numpy()
        return np.random.choice(2, size=(probs.shape[0], num_samples), p=probs)
