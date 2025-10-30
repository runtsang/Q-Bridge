import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class LayerParams:
    """Parameters for the clipping‑scale head."""
    scale: float = 1.0
    shift: float = 0.0
    clip: bool = True

class ClippedLinear(nn.Module):
    """Linear layer with optional clipping and affine post‑processing."""
    def __init__(self, in_features: int, out_features: int, params: LayerParams) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("scale", torch.tensor(params.scale))
        self.register_buffer("shift", torch.tensor(params.shift))
        self.clip = params.clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.clip:
            out = torch.clamp(out, -5.0, 5.0)
        return out * self.scale + self.shift

class QuantumHybridModel(nn.Module):
    """Classical CNN + FC backbone with a QCNN‑style quantum head."""
    def __init__(self) -> None:
        super().__init__()
        # CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully‑connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        # Quantum layer placeholder – set by the QML module
        self.quantum_layer = None  # type: ignore
        # Classical head with clipping & scaling
        self.head = ClippedLinear(32, 1, LayerParams(scale=1.0, shift=0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        proj = self.fc(flat)
        if self.quantum_layer is not None:
            proj = self.quantum_layer(proj)
        out = self.head(proj)
        return torch.sigmoid(out).squeeze(-1)

__all__ = ["QuantumHybridModel"]
