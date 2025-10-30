import torch
from torch import nn
import torch.nn.functional as F

def _build_classifier_circuit(num_features: int, depth: int) -> nn.Module:
    """
    Build a small feed‑forward network that mirrors a quantum classifier.
    Adds dropout after each ReLU for regularisation.
    """
    layers = []
    in_dim = num_features
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.1))
    head = nn.Linear(num_features, 2)
    layers.append(head)
    return nn.Sequential(*layers)

class ConvGen078(nn.Module):
    """
    Hybrid classical convolution + classifier that can replace the original Conv filter.
    Parameters
    ----------
    kernel_size : int
        Size of the 2‑D filter (default 2).
    threshold : float
        Activation threshold for the convolution output.
    depth : int
        Depth of the classifier network (default 1).
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, depth: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        # classifier input dimension matches number of pixels in the patch
        num_features = kernel_size ** 2
        self.classifier = _build_classifier_circuit(num_features, depth)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolution followed by the classifier.
        """
        conv_out = self.conv(data)
        activations = torch.sigmoid(conv_out - self.threshold)
        # flatten to (batch, num_features)
        features = activations.view(activations.size(0), -1)
        logits = self.classifier(features)
        return logits

    def run(self, data) -> torch.Tensor:
        """
        Convenience wrapper that accepts a NumPy array or torch tensor.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        # Ensure shape (1, 1, H, W)
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndim == 3 and data.shape[0] == 1:
            data = data.unsqueeze(1)
        logits = self.forward(data)
        probs = F.softmax(logits, dim=-1)
        return probs.squeeze()

__all__ = ["ConvGen078"]
