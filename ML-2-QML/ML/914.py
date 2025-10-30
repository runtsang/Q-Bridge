import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """A simple residual block to improve gradient flow for deeper nets."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)) + x)

class TemperatureHybrid(nn.Module):
    """Hybrid head with learnable temperature for controlled softmax."""
    def __init__(self, in_features: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x) / self.temperature
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

class QuantumHybridClassifier(nn.Module):
    """Classical CNN with a temperature‑controlled hybrid head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.residual = ResidualBlock(15)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = TemperatureHybrid(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.residual(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.head(x)

    def fit(self,
            dataloader,
            optimizer,
            loss_fn=nn.BCELoss(),
            epochs: int = 5,
            device: str = "cpu") -> None:
        """Simple training loop for the hybrid classifier."""
        self.to(device)
        self.train()
        for epoch in range(epochs):
            for data, target in dataloader:
                data, target = data.to(device), target.to(device).float()
                optimizer.zero_grad()
                probs = self(data)
                loss = loss_fn(probs, target)
                loss.backward()
                optimizer.step()

    def predict(self, data: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        """Return class probabilities for the given data."""
        self.eval()
        with torch.no_grad():
            return self(data.to(device))
