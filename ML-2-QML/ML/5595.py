import numpy as np
import torch
from torch import nn
from torch.nn.functional import sigmoid, tanh

class HybridFCL(nn.Module):
    """
    Hybrid fully‑connected layer that supports three operational modes:

    * ``classical`` – a standard linear layer followed by a tanh non‑linearity.
    * ``kernel`` – interprets the linear weights as support vectors and
      evaluates an RBF kernel between the input and the weights.
    * ``conv`` – a 2‑D convolutional filter that mimics a quanvolution
      layer; the convolution output is passed through a sigmoid and
      averaged.

    The ``run`` method keeps the same signature as the original FCL
    example, returning a NumPy array containing the expectation value.
    """

    def __init__(self,
                 n_features: int = 1,
                 mode: str = "classical",
                 kernel_gamma: float = 1.0,
                 conv_kernel_size: int = 2,
                 conv_threshold: float = 0.0):
        super().__init__()
        self.mode = mode
        self.n_features = n_features

        if mode == "classical":
            self.linear = nn.Linear(n_features, 1)
        elif mode == "kernel":
            # store support vectors in a linear layer (bias disabled)
            self.linear = nn.Linear(n_features, 1, bias=False)
            self.gamma = kernel_gamma
        elif mode == "conv":
            self.conv = nn.Conv2d(1, 1, kernel_size=conv_kernel_size, bias=True)
            self.threshold = conv_threshold
        else:
            raise ValueError(f"unknown mode {mode}")

    def run(self, thetas: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.mode == "classical":
            values = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
            return tanh(self.linear(values)).mean(dim=0).detach().numpy()

        if self.mode == "kernel":
            support = self.linear.weight.view(-1)
            vec = torch.as_tensor(thetas, dtype=torch.float32).view(-1)
            diff = support - vec
            k = torch.exp(-self.gamma * diff.pow(2).sum())
            return k.detach().numpy()

        if self.mode == "conv":
            tensor = torch.as_tensor(thetas, dtype=torch.float32)
            tensor = tensor.view(1, 1, *tensor.shape)
            logits = self.conv(tensor)
            activations = sigmoid(logits - self.threshold)
            return np.array([activations.mean().item()])

        raise RuntimeError("unreachable")

__all__ = ["HybridFCL"]
