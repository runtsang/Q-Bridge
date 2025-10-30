import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuanvolutionHybrid(nn.Module):
    """Classical hybrid model with a 2×2 convolutional backbone and a
    fully‑connected layer that mimics a quantum kernel."""
    def __init__(self, n_channels: int = 1, n_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_channels, 4, kernel_size=2, stride=2)
        self.linear = nn.Linear(4 * 14 * 14, n_classes)
        # dummy quantum‑style fully connected layer
        self.qfc = nn.Linear(4 * 14 * 14, 4 * 14 * 14)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        flat = features.view(x.size(0), -1)
        # simulate quantum expectation via tanh of a linear transform
        q_features = torch.tanh(self.qfc(flat))
        logits = self.linear(q_features)
        return F.log_softmax(logits, dim=-1)

    def run_qfc(self, thetas: np.ndarray) -> np.ndarray:
        """Mimic a quantum fully‑connected layer using a classical linear
        transform followed by tanh activation. Thetas are interpreted as
        parameters of the linear layer."""
        with torch.no_grad():
            params = torch.tensor(thetas, dtype=torch.float32).view(-1, 1)
            out = torch.tanh(self.qfc.weight @ params).mean()
        return out.cpu().numpy()

__all__ = ["QuanvolutionHybrid"]
