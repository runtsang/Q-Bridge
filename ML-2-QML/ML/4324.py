import torch
import torch.nn as nn
import numpy as np

# ----------------------------------------------------------------------
# Classical stand‑ins for the quantum modules used in the original seed
# ----------------------------------------------------------------------
def Conv():
    """Return a callable object that emulates the quantum filter with PyTorch ops."""
    class ConvFilter(nn.Module):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(self, data) -> float:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()
    return ConvFilter()

def FCL():
    """Return an object with a ``run`` method mimicking the quantum example."""
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: np.ndarray) -> np.ndarray:
            values = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()
    return FullyConnectedLayer()
# ----------------------------------------------------------------------
# Main hybrid model
# ----------------------------------------------------------------------
class QFCLayer(nn.Module):
    """
    Classical surrogate for the quantum fully‑connected layer.
    Applies a linear map followed by tanh and a per‑sample mean.
    """
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        out = torch.tanh(self.linear(x))
        # per‑sample mean across the feature dimension
        return out.mean(dim=1, keepdim=True)

class HybridNAT(nn.Module):
    """
    Classical CNN + quantum‑inspired fully‑connected layer + classical convolution filter.
    """
    def __init__(self) -> None:
        super().__init__()
        # CNN backbone (from reference 1)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classical quantum‑inspired fully‑connected layer
        self.qfc = QFCLayer(16 * 7 * 7)
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)
        # Classical convolution filter (from reference 3)
        self.conv_filter = Conv()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)            # (bs, 16, 7, 7)
        flat = feat.view(bsz, -1)          # (bs, 16*7*7)
        qfc_out = self.qfc(flat)           # (bs, 1)

        # Compute the classical convolution filter on the *raw* image
        conv_feat = torch.tensor(
            [self.conv_filter.run(x[i, 0].cpu().numpy()) for i in range(bsz)],
            device=x.device
        ).unsqueeze(1)                      # (bs, 1)

        # Concatenate the quantum‑inspired scalar with the flattened features
        flat = torch.cat([flat, qfc_out, conv_feat], dim=1)  # (bs, 16*7*7 + 2)
        out = self.classifier(flat)
        return self.norm(out)

__all__ = ["HybridNAT"]
