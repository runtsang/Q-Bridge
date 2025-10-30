import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFilter(nn.Module):
    """Classical 2‑D filter emulating a quanvolution layer."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 1, H, W)

        Returns
        -------
        torch.Tensor
            Shape (B, 1) – mean activation over the kernel window.
        """
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])


class SamplerQNN(nn.Module):
    """
    Classical sampler network that receives the output of ConvFilter
    and produces a probability distribution over two classes.
    """

    def __init__(self, kernel_size: int = 2) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=kernel_size)
        # The sampler network is deliberately shallow to mirror the QNN
        self.net = nn.Sequential(
            nn.Linear(1, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (B, 1, H, W)

        Returns
        -------
        torch.Tensor
            Shape (B, 2) – softmax probabilities.
        """
        features = self.conv(inputs).view(inputs.size(0), -1)
        logits = self.net(features)
        return F.softmax(logits, dim=-1)


def SamplerQNN():
    """Return a ready‑to‑use instance of the hybrid classical sampler."""
    return SamplerQNN()


__all__ = ["SamplerQNN"]
