import torch
import torch.nn as nn

class ClassicalFilter(nn.Module):
    """Classical convolutional filter emulating the quantum filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        # Return mean activation per image
        return activations.mean(dim=[2, 3]).unsqueeze(1)

class HybridNATModel(nn.Module):
    """
    Hybrid classical model that augments a convolutional feature extractor
    with a quantum-inspired filter and a variational linear layer.
    """
    def __init__(self, conv_kernel: int = 2, conv_threshold: float = 0.0) -> None:
        super().__init__()
        self.quantum_filter = ClassicalFilter(kernel_size=conv_kernel, threshold=conv_threshold)

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Classical approximation of the quantum variational layer
        self.q_layer = nn.Sequential(
            nn.Linear(16 * 7 * 7, 4),
            nn.Tanh()
        )

        # Final fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7 + 1 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantum‑inspired filter output
        filter_out = self.quantum_filter(x)          # (batch, 1)

        # Classical feature extractor
        features = self.features(x).view(x.shape[0], -1)  # (batch, 784)

        # Classical variational layer
        q_out = self.q_layer(features)                # (batch, 4)

        # Concatenate all signals
        concat = torch.cat([features, filter_out, q_out], dim=1)

        out = self.fc(concat)
        return self.norm(out)

__all__ = ["HybridNATModel"]
