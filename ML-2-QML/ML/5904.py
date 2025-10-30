import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuantumNatNet(nn.Module):
    """
    Classical hybrid network inspired by Quantum‑NAT and hybrid binary classifiers.
    It contains a CNN backbone, a fully‑connected projection, and a
    trainable hybrid head that can emulate a quantum expectation value
    using a parameter‑shift style sigmoid.
    """

    def __init__(self,
                 in_channels: int = 1,
                 use_quantum_head: bool = False,
                 shift: float = 0.0,
                 device: str = "cpu"):
        super().__init__()
        # CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classical fully‑connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

        # Hybrid head
        self.head = nn.Linear(4, 1)  # maps 4 features to a scalar logit
        self.shift = shift
        self.use_quantum_head = use_quantum_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that optionally mimics a quantum expectation head
        via a learnable linear layer followed by a sigmoid with shift.
        """
        features = self.backbone(x)
        flattened = features.view(x.size(0), -1)
        out = self.fc(flattened)
        out = self.norm(out)

        # Classical head
        logits = self.head(out)
        probs = torch.sigmoid(logits + self.shift)

        # If a quantum head is desired, the user can replace `self.head`
        # with a quantum module in the qml implementation.
        return probs

__all__ = ["HybridQuantumNatNet"]
