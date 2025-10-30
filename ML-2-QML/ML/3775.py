import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumInspiredFC(nn.Module):
    """Classical emulation of a quantum fully‑connected layer."""
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.tanh(self.linear(x))
        return y.mean(dim=1, keepdim=True)

class HybridFCQuanvolution(nn.Module):
    """
    Hybrid classical‑quantum architecture that combines a classical
    convolutional filter (inspired by the Quanvolution example) with a
    quantum‑inspired fully‑connected layer (inspired by the FCL example).
    """
    def __init__(self) -> None:
        super().__init__()
        # Classical convolution (2×2 kernel, stride 2)
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Flatten and quantum‑inspired fully‑connected layer
        self.quantum_fc = QuantumInspiredFC(4 * 14 * 14)
        # Final linear head to 10 classes
        self.linear = nn.Linear(1, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        flat = features.view(x.size(0), -1)
        qfc_out = self.quantum_fc(flat)
        logits = self.linear(qfc_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridFCQuanvolution"]
