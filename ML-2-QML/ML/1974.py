import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SEBlock(nn.Module):
    """Squeeze‑and‑Excitation block for channel‑wise attention."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class QuantumNATExtended(nn.Module):
    """
    Classical hybrid model inspired by Quantum‑Nat, extended with:
    • Residual CNN blocks (two conv layers each) for richer feature extraction.
    • A squeeze‑and‑excitation module for channel‑wise attention.
    • A lightweight MLP classifier producing four outputs.
    """
    def __init__(self, in_channels: int = 1, n_heads: int = 4, depth: int = 2):
        super().__init__()
        # Residual blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
        )
        self.pool = nn.MaxPool2d(2)
        self.se = SEBlock(16)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = self.block1(x)
        x = x + x  # Residual connection (identity)
        x = self.pool(x)
        x = self.block2(x)
        x = self.se(x)
        x = x + x  # Residual connection (identity)
        x = self.pool(x)
        x_flat = x.view(bsz, -1)  # (B, 784)
        logits = self.classifier(x_flat)
        return self.bn(logits)

    def fit(self, data_loader, lr: float = 1e-3, epochs: int = 10, device: str = 'cpu'):
        """Simple training loop for quick experimentation."""
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.train()
        for epoch in range(epochs):
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = self(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
