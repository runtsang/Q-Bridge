import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class QuanvolutionFilter(nn.Module):
    """
    Classical convolutional filter inspired by the original quanvolution example.
    Supports multiple output channels and configurable kernel size.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class QuanvolutionClassifier(nn.Module):
    """
    Hybrid neural network using the quanvolutional filter followed by a linear head.
    Provides fit and predict helpers for quick experimentation.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels, out_channels)
        self.fc = nn.Linear(out_channels * 14 * 14, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qfilter(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)
    def fit(self, train_loader: DataLoader, epochs: int = 5, lr: float = 1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.NLLLoss()
        self.train()
        for epoch in range(epochs):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp().argmax(dim=-1)
