import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridNatRegression(nn.Module):
    """
    Classical hybrid model that takes an image and optional quantum features.
    Combines the convolutional front‑end of QuantumNAT with a regression head
    inspired by EstimatorQNN.  When quantum features are provided the model
    learns to fuse them with classical representations for regression.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        # Classical CNN front‑end
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classical fully‑connected projection (original 4‑dim output)
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)
        # Regression head that fuses classical and quantum features
        self.regressor = nn.Sequential(
            nn.Linear(4 + n_qubits, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor, quantum_feat: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input image batch of shape (B, 1, H, W).
            quantum_feat: Optional tensor of shape (B, n_qubits) produced by a quantum encoder.
        Returns:
            If quantum_feat is provided, a scalar regression output of shape (B, 1).
            Otherwise, the 4‑dim classical projection.
        """
        bsz = x.shape[0]
        feat = self.features(x).view(bsz, -1)
        out = self.fc(feat)
        out = self.norm(out)
        if quantum_feat is not None:
            combined = torch.cat([out, quantum_feat], dim=1)
            out = self.regressor(combined)
        return out

__all__ = ["HybridNatRegression"]
