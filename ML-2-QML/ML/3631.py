import torch
import torch.nn as nn

class HybridNATModel(nn.Module):
    """Hybrid classical‑only model that mimics a quantum‑style feature map.

    The network combines a 2‑D CNN feature extractor, a simple
    variational‑style linear block that emulates the parameterised
    rotations of a quantum circuit, and a final regression head.
    """
    def __init__(self, input_channels: int = 1, output_features: int = 1) -> None:
        super().__init__()
        # Feature extractor (from QuantumNAT)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Simulated quantum layer: three trainable rotations
        self.quantum_sim = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        # Regression head (EstimatorQNN style)
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_features)
        )
        self.norm = nn.BatchNorm1d(output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        f = self.features(x)
        f_flat = f.view(bsz, -1)
        q_out = self.quantum_sim(f_flat)
        pred = self.head(q_out)
        return self.norm(pred)

__all__ = ["HybridNATModel"]
