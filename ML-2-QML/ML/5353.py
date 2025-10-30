import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2,3])

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class HybridSamplerQNN(nn.Module):
    """
    Classical sampler network that emulates a hybrid quantum sampler.
    Combines a 2‑D convolution filter, a hybrid quantum‑like head and
    a probabilistic sampler.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 shift: float = 0.0, num_samples: int = 10):
        super().__init__()
        self.conv = ConvFilter(kernel_size, threshold)
        self.hybrid = Hybrid(1, shift)
        self.num_samples = num_samples

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape (batch, 1, H, W)
        Returns:
            probs: Tensor of shape (batch, 2) with class probabilities
        """
        features = self.conv(inputs)  # shape (batch, 1)
        logits = self.hybrid(features)  # shape (batch, 1)
        probs = torch.cat((logits, 1 - logits), dim=-1)
        return probs

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the
        network output.
        """
        probs = self.forward(inputs)  # (batch, 2)
        return torch.multinomial(probs, self.num_samples, replacement=True)

__all__ = ["HybridSamplerQNN"]
