import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumFullyConnected(nn.Module):
    """Classical surrogate of a quantum fully‑connected layer.
    Uses a linear transform followed by a tanh non‑linearity to mimic
    the expectation value produced by a parameterised Ry+CX circuit."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # thetas shape: (batch, n_features)
        expectation = torch.tanh(self.linear(thetas)).mean(dim=0, keepdim=True)
        return expectation

class QuanvolutionFilter(nn.Module):
    """Classical 2‑D convolutional filter inspired by the quantum quanvolution.
    Operates on 28×28 single‑channel images with a 2×2 kernel and stride 2,
    producing 14×14 feature maps that are flattened."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class SamplerQNN(nn.Module):
    """Purely classical sampler that emulates a quantum sampler.
    A small feed‑forward network ending with a softmax that mimics
    the probability distribution obtained from a quantum statevector sampler."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class HybridQuantumNet(nn.Module):
    """
    Hybrid architecture combining:
      * QuanvolutionFilter – classical analogue of a quantum convolution
      * QuantumFullyConnected – classical surrogate of a quantum fully‑connected layer
      * SamplerQNN – classical approximation of a quantum sampler
      * Linear head – final classification layer
    The network is fully differentiable and can be trained with standard
    back‑propagation.  It mirrors the structure of the quantum‑inspired
    reference implementations while remaining purely classical.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.qfc = QuantumFullyConnected(n_features=4 * 14 * 14)
        self.sampler = SamplerQNN()
        self.classifier = nn.Linear(4 * 14 * 14 + 1 + 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        # 1. Quanvolution feature extraction
        qfeat = self.qfilter(x)  # (batch, 4*14*14)

        # 2. Quantum fully‑connected layer (classical surrogate)
        qfc_out = self.qfc(qfeat)  # (1,)

        # 3. Quantum sampler – sample from 2‑dimensional distribution
        sampler_in = qfeat[:, :2]
        sampler_out = self.sampler(sampler_in)  # (batch, 2)

        # 4. Concatenate all representations
        combined = torch.cat([qfeat, qfc_out.expand_as(qfeat[:, :1]), sampler_out], dim=1)

        # 5. Classification head
        logits = self.classifier(combined)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuantumFullyConnected", "QuanvolutionFilter", "SamplerQNN", "HybridQuantumNet"]
