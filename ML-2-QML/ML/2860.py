import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class QuantumFeatureExtractor(nn.Module):
    """
    Classical emulation of a 2‑qubit SamplerQNN circuit.
    Parameters are treated as rotation angles; the forward pass
    produces a 2‑dimensional probability vector that mimics the
    quantum sampler output.
    """
    def __init__(self, clip: bool = True):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(4))
        self.scale = nn.Parameter(torch.ones(2))
        self.shift = nn.Parameter(torch.zeros(2))
        self.clip = clip

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.clip:
            w = torch.clamp(self.weights, -5.0, 5.0)
        else:
            w = self.weights
        theta0 = inputs[..., 0] + w[0] + w[2]
        theta1 = inputs[..., 1] + w[1] + w[3]
        amp0 = torch.cos(theta0)
        amp1 = torch.cos(theta1)
        amp2 = torch.sin(theta0)
        amp3 = torch.sin(theta1)
        probs = torch.stack([amp0**2 + amp2**2,
                             amp1**2 + amp3**2], dim=-1)
        probs = probs * self.scale + self.shift
        probs = F.softmax(probs, dim=-1)
        return probs

class SamplerQNNHybrid(nn.Module):
    """
    Hybrid sampler network that combines a classical feature extractor,
    a quantum‑inspired feature layer, and a final classifier.
    """
    def __init__(self, clip: bool = True):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        self.quantum_layer = QuantumFeatureExtractor(clip=clip)
        self.classifier = nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_net(inputs)
        q = self.quantum_layer(x)
        logits = self.classifier(q)
        return logits

__all__ = ["SamplerQNNHybrid", "QuantumFeatureExtractor", "FraudLayerParameters"]
