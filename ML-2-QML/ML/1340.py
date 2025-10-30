import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

class HybridFunction(nn.Module):
    """Sigmoid activation with a learnable shift."""
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = nn.Parameter(torch.tensor(shift))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)

class HybridBinaryClassifier(nn.Module):
    """Classical CNN with a learnable dense head and optional calibration."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        # Flatten and dense head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.activation = HybridFunction()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class indices."""
        probs = self.forward(x)
        return torch.argmax(probs, dim=-1)

    def calibrate(self, probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE)."""
        probs = probs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
            if np.any(mask):
                bin_acc = np.mean(targets[mask] == np.argmax(probs[mask], axis=1))
                bin_conf = np.mean(probs[mask])
                ece += np.abs(bin_conf - bin_acc) * np.sum(mask) / len(probs)
        return ece

    def early_stopping(self, val_loss: float, best_loss: float, patience: int, counter: int) -> tuple[bool, int, float]:
        """Simple early stopping logic."""
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
        stop = counter >= patience
        return stop, counter, best_loss

__all__ = ["HybridBinaryClassifier", "HybridFunction"]
