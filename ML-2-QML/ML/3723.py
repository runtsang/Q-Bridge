import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation mimicking a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float = 0.0):
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Classical head replacing the quantum layer."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class HybridQCNNClassifier(nn.Module):
    """Hybrid model with a QCNN‑style quantum head and a classical convolutional backbone."""
    def __init__(self, in_channels: int = 3, num_qubits: int = 8, shots: int = 100, shift: float = np.pi/2):
        super().__init__()
        # Classical convolutional front‑end
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully connected layers producing 8 features for the QCNN input
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)
        # Classical hybrid head
        self.hybrid = Hybrid(8, shift=shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x).squeeze()
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQCNNClassifier", "Hybrid", "HybridFunction"]
