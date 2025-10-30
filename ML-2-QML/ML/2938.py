import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """Classical 2‑D convolutional filter emulating a quantum quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Return a scalar activation for a single kernel window."""
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

class SamplerQNN(nn.Module):
    """Simple feed‑forward sampler network producing a 2‑class probability."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class HybridSamplerConv(nn.Module):
    """Hybrid classical‑quantum sampler that first applies a ConvFilter then a SamplerQNN."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size, threshold)
        self.sampler = SamplerQNN()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Input window of shape (kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (2,).
        """
        conv_feat = self.conv(data)          # scalar
        mean_feat = data.mean()              # second feature
        inputs = torch.stack([conv_feat, mean_feat], dim=-1)
        return self.sampler(inputs)

__all__ = ["HybridSamplerConv"]
