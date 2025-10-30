import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerConv(nn.Module):
    """
    A hybrid module that combines a two‑class sampler network with a 2‑D convolutional filter.
    The sampler produces a probability distribution over two outputs, while the convolution
    processes a single‑channel image. The final scalar is the product of the mean activation
    from the convolution and the sum of sampler probabilities, providing a simple yet
    expressive feature representation.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        # Sampler part: 2‑input → 4 hidden → 2‑output, Tanh non‑linearity
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )
        # Convolutional filter with learnable bias
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.threshold = threshold

    def forward(self, inputs: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, 2) fed into the sampler.
        image : torch.Tensor
            Tensor of shape (batch, 1, H, W) fed into the convolution.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch,) containing the combined scalar output.
        """
        # Sampler output: probability distribution over two classes
        sampler_out = F.softmax(self.sampler(inputs), dim=-1)

        # Convolution output processed by a sigmoid activation with a threshold
        conv_out = self.conv(image)
        activation = torch.sigmoid(conv_out - self.threshold)
        conv_score = activation.mean(dim=(2, 3))

        # Combine: mean convolution score multiplied by the sum of sampler probabilities
        return conv_score * sampler_out.sum(dim=-1)

__all__ = ["HybridSamplerConv"]
