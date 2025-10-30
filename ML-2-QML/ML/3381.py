import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridFunction(nn.Module):
    """Custom activation blending linear and sigmoid behaviour."""
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = nn.Parameter(torch.tensor(shift))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x + self.shift)

class HybridHead(nn.Module):
    """Small MLP ending with HybridFunction for binary logits."""
    def __init__(self, in_features: int, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)
        self.hf = HybridFunction()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return self.hf(x)

class CNNBackbone(nn.Module):
    """Feature extractor with two conv layers and pooling."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop = nn.Dropout(0.3)
        self.flatten_size = 32 * 4 * 4  # assumes 32×32 input
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop(x)
        return torch.flatten(x, 1)

class ClassicalQLSTM(nn.Module):
    """Classical LSTM cell for sequence tagging."""
    def __init__(self, input_dim: int, hidden_dim: int, tagset: int = 10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset)
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq)
        return F.log_softmax(self.fc(out), dim=-1)

class HybridBinaryQLSTMNet(nn.Module):
    """Combined CNN + Hybrid head + optional Classical LSTM."""
    def __init__(self, use_lstm: bool = False, lstm_input: int = 32*4*4, lstm_hidden: int = 64):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = HybridHead(self.backbone.flatten_size)
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = ClassicalQLSTM(lstm_input, lstm_hidden)
    def forward(self, image: torch.Tensor, seq: torch.Tensor | None = None) -> torch.Tensor | tuple:
        feat = self.backbone(image)
        logits = self.head(feat)
        probs = torch.cat((logits, 1 - logits), dim=-1)
        if self.use_lstm and seq is not None:
            tag_out = self.lstm(seq)
            return probs, tag_out
        return probs

__all__ = ["HybridBinaryQLSTMNet", "HybridHead", "HybridFunction", "CNNBackbone", "ClassicalQLSTM"]
