import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridFunction(torch.autograd.Function):
    """
    Differentiable hybrid head that applies a sigmoid with a learnable shift.
    The shift parameter can be fixed or trainable, enabling richer activation dynamics.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None

class Hybrid(nn.Module):
    """
    Hybrid dense layer that can operate in two modes:
        1. Linear + sigmoid with optional trainable shift
        2. Small MLP gating mechanism that mixes linear and nonlinear outputs.
    """
    def __init__(self, in_features: int, use_mlp: bool = False, shift_learnable: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = nn.Parameter(torch.zeros(1)) if shift_learnable else torch.tensor(0.0)
        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(in_features, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            self.gate = nn.Linear(in_features, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        if self.use_mlp:
            mlp_out = self.mlp(logits)
            gate = torch.sigmoid(self.gate(logits))
            out = gate * mlp_out + (1 - gate) * self.linear(logits)
            return HybridFunction.apply(out, self.shift)
        else:
            return HybridFunction.apply(self.linear(logits), self.shift)

class QCNet(nn.Module):
    """
    CNN-based binary classifier with optional quantum hybrid head.
    The architecture mirrors the original but adds batch normalization,
    dropout, and an sklearn-compatible wrapper for training pipelines.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(15)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(self.fc3.out_features, use_mlp=True, shift_learnable=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), dim=-1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning class logits."""
        return self.forward(X)

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 10,
            lr: float = 1e-3, device: str = "cpu") -> None:
        """Simple training loop exposing a sklearn-like fit API."""
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        X, y = X.to(device), y.to(device)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            logits = self.forward(X).squeeze(-1)
            loss = criterion(logits, y.float())
            loss.backward()
            optimizer.step()
            if epoch % (epochs // 5 + 1) == 0:
                print(f"Epoch {epoch+1}/{epochs} loss: {loss.item():.4f}")

__all__ = ["HybridFunction", "Hybrid", "QCNet"]
