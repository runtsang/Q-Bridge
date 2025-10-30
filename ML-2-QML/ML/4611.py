import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybridNet(nn.Module):
    """Classical hybrid network combining a convolutional feature extractor,
    an LSTM for sequential modeling, and a linear classifier.
    It mirrors the quantum counterpart while staying fully classical."""
    def __init__(self, input_channels: int = 3, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Compute flatten size
        dummy = torch.zeros(1, input_channels, 32, 32)
        x = F.relu(self.conv1(dummy))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        self.flatten_size = x.numel()

        # LSTM for sequential modeling
        self.lstm = nn.LSTM(self.flatten_size, hidden_dim, batch_first=True)

        # Classifier
        self.fc1 = nn.Linear(hidden_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        # LSTM expects (batch, seq_len, features)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        hidden = lstm_out.squeeze(1)
        x = F.relu(self.fc1(hidden))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybridNet"]
