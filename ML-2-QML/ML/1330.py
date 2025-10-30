import torch
import torch.nn as nn

class HybridQuantumBinaryClassifier(nn.Module):
    """
    Classical baseline binary classifier that mimics the architecture of the hybrid quantum model.
    The head is a simple MLP that produces a probability for binary classification.
    """
    def __init__(self, in_features: int = None, hidden: int = 64, shift: float = 0.0):
        super().__init__()
        # Feature extractor identical to the quantum version
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
            nn.Flatten()
        )
        # Determine the size of the flattened feature map
        if in_features is None:
            dummy = torch.zeros(1, 3, 32, 32)
            out = self.feature_extractor(dummy)
            in_features = out.shape[1]
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.fc(features)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat([probs, 1 - probs], dim=-1)
