import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid with optional shift."""
    @staticmethod
    def forward(ctx, inputs, shift):
        out = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        return grad_output * out * (1 - out), None

class Hybrid(nn.Module):
    """Dense head applying HybridFunction."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(self.linear(x).view(-1), self.shift)

class ConvFilter(nn.Module):
    """Classical convolutional filter emulating quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.threshold = threshold

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.kernel(data)
        return torch.sigmoid(logits - self.threshold).mean()

class SamplerModule(nn.Module):
    """Softmax sampler mirroring QNN SamplerQNN."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

def build_classifier_circuit(num_features: int, depth: int):
    """
    Factory that returns a classical classifier mirroring the quantum interface.
    The returned tuple contains:
        - nn.Module: the full network
        - encoding: list of input feature indices
        - weight_sizes: list of parameter counts per layer
        - observables: placeholder list for compatibility
    """
    layers = []

    # Convolutional backbone
    layers.append(nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 1))
    layers.append(nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 1))
    layers.append(nn.Flatten())

    # Fully‑connected head
    layers.append(nn.Linear(55815, 120))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(120, 84))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(84, num_features))
    layers.append(Hybrid(num_features))

    net = nn.Sequential(*layers)

    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in net.parameters()]
    observables = [0, 1]  # placeholder for quantum‑style observables

    return net, encoding, weight_sizes, observables

__all__ = ["HybridFunction", "Hybrid", "ConvFilter", "SamplerModule", "build_classifier_circuit"]
