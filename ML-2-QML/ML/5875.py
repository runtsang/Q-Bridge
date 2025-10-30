"""
HybridClassifier: Classical CNN backbone with a learnable MLP head.

This module mirrors the original QCNet but replaces the quantum
expectation layer with a small, fully‑connected network that is
trained jointly with the rest of the model.  The head can be
used for ablation studies against the quantum version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridClassifier(nn.Module):
    """
    Classical CNN backbone followed by a trainable MLP head.
    """
    def __init__(self, n_classes: int = 2, hidden_dim: int = 32):
        super().__init__()
        # Convolutional backbone (identical to the original QCNet)
        self.backbone = nn.Sequential(
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
        # Fully connected layers from the seed
        self.fc1 = nn.Linear(55815, 120)   # 55815 = output size after flatten
        self.fc2 = nn.Linear(120, 84)
        # Learnable MLP head
        self.head = nn.Sequential(
            nn.Linear(84, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        logits = self.head(x)
        return F.softmax(logits, dim=-1)  # shape: (batch, n_classes)
