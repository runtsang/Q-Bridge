import torch
from torch import nn
import torch.nn.functional as F

class ConvGen189(nn.Module):
    """Hybrid classical model combining convolution, fraud‑style fully‑connected layers,
    a quantum‑inspired projection, and a sampler‑style classifier.

    The architecture mirrors the seed Conv filter, expands it with a
    multi‑layer ConvNet (QuantumNAT), injects FraudDetection style
    linear transformations, and ends with a lightweight SamplerQNN
    to produce a 2‑class probability distribution.
    """

    def __init__(self,
                 conv_kernel: int = 3,
                 conv_features: int = 8,
                 fraud_features: int = 2,
                 sampler_features: int = 4):
        super().__init__()
        # Classical convolutional backbone (QuantumNAT style)
        self.features = nn.Sequential(
            nn.Conv2d(1, conv_features, kernel_size=conv_kernel, padding=conv_kernel//2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_features, conv_features*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # FraudDetection‑style linear block
        self.fraud = nn.Sequential(
            nn.Linear(2, fraud_features),
            nn.Tanh(),
            nn.Linear(fraud_features, fraud_features),
            nn.Tanh()
        )
        # Quantum‑inspired fully‑connected projection
        self.qfc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, sampler_features)
        )
        # SamplerQNN classifier
        self.sampler = nn.Sequential(
            nn.Linear(sampler_features, 2),
            nn.Tanh(),
            nn.Linear(2, 2)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # feed a 2‑dimensional projection to fraud block
        x_proj = x[:, :2]
        x = self.fraud(x_proj)
        # quantum‑inspired projection
        x = self.qfc(x)
        x = self.sampler(x)
        return self.softmax(x)

__all__ = ["ConvGen189"]
