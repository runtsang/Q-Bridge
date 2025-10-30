import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFunction(torch.autograd.Function):
    """Differentiable approximation of a quantum expectation head using a parameter‑shift rule."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.save_for_backward(inputs)
        # Toy quantum expectation: cosine of the input
        return torch.cos(inputs)
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        grad_inputs = grad_output * (-torch.sin(inputs))
        return grad_inputs, None

class QuantumEnhancedHybridNet(nn.Module):
    """Classical CNN backbone with a hybrid quantum‑like activation head."""
    def __init__(self, shift: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
        self.shift = shift
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x).squeeze(-1)
        probs = HybridFunction.apply(x, self.shift)
        return torch.stack([probs, 1 - probs], dim=-1)
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    @staticmethod
    def train_step(model: "QuantumEnhancedHybridNet",
                   dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module,
                   device: torch.device) -> float:
        model.train()
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        return total_loss / len(dataloader.dataset)

__all__ = ["HybridFunction", "QuantumEnhancedHybridNet"]
