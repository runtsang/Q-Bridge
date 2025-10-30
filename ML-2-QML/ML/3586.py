import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFCL(nn.Module):
    """Classical fully connected layer with quantum‑inspired activation.

    The module follows the architecture of the Quantum‑NAT CNN backbone
    but replaces the final scalar activation with the expectation value of
    a single‑qubit Ry circuit, i.e. cos(θ).  This keeps the forward pass
    entirely classical while still reflecting a quantum measurement
    result.  Batch‑normalisation is applied to the four‑dimensional output
    to stabilise training.
    """
    def __init__(self) -> None:
        super().__init__()
        # CNN feature extractor (identical to QFCModel in Quantum‑NAT)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        # Quantum‑inspired activation and normalisation
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        flattened = features.view(x.size(0), -1)
        linear_out = self.fc(flattened)
        # quantum‑inspired activation: expectation of Pauli‑Z after Ry(θ) = cos(θ)
        quantum_act = torch.cos(linear_out)
        out = self.norm(quantum_act)
        return out

__all__ = ["HybridFCL"]
