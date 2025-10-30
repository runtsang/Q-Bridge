import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the quantum head from the separate QML module
from.QuantumHead import QuantumHead

class HybridQCNet(nn.Module):
    """Hybrid CNN + 4‑qubit variational quantum head for binary classification."""
    def __init__(self, in_channels: int = 3, num_qubits: int = 4) -> None:
        super().__init__()
        # Feature extractor (inspired by Quantum‑NAT)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Projection to num_qubits features
        self.fc = nn.Sequential(
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_qubits),
        )
        # Quantum head
        self.quantum_head = QuantumHead(n_wires=num_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        q_out = self.quantum_head(x)           # shape (batch, 1)
        logits = q_out.squeeze(-1)            # shape (batch,)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridQCNet"]
