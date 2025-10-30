import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumExpectationLayer(torch.autograd.Function):
    """Approximate a quantum expectation using a sinusoidal activation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        return torch.sin(inputs)
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        inputs, = ctx.saved_tensors
        return grad_output * torch.cos(inputs)

class SamplerModule(nn.Module):
    """Classical network mimicking a quantum sampler."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class HybridHead(nn.Module):
    """Hybrid head that can be either a classic expectation mimic or a sampler."""
    def __init__(self, method: str = "expectation", shift: float = np.pi/2) -> None:
        super().__init__()
        self.method = method
        if method == "sampler":
            self.sampler = SamplerModule()
        else:
            self.linear = nn.Linear(1, 1)
            self.shift = shift
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.method == "sampler":
            return self.sampler(inputs)
        logits = self.linear(inputs)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

class HybridBinaryClassifier(nn.Module):
    """CNN backbone followed by a hybrid head for binary classification."""
    def __init__(self, head_method: str = "expectation") -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1)
        )
        self.head = HybridHead(method=head_method)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.backbone(inputs)
        return self.head(x)
