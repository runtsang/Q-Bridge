import torch
from torch import nn
import torch.nn.functional as F

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.inputs = inputs
        exp_vals = circuit.run(inputs.tolist())
        exp_tensor = torch.tensor(exp_vals, dtype=torch.float32, device=inputs.device)
        return exp_tensor

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.inputs
        shift = ctx.shift
        grad_inputs = []
        for val in inputs.cpu().numpy():
            right = ctx.circuit.run([val + shift])[0]
            left = ctx.circuit.run([val - shift])[0]
            grad_inputs.append(right - left)
        grad_tensor = torch.tensor(grad_inputs, dtype=torch.float32, device=inputs.device)
        return grad_tensor * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid head that optionally uses a quantum circuit."""
    def __init__(self, in_features: int, circuit=None, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.circuit = circuit
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x).squeeze(-1)
        if self.circuit is None:
            return torch.sigmoid(logits)
        return HybridFunction.apply(logits, self.circuit, self.shift)

class QCNNBackbone(nn.Module):
    """Convolutional backbone inspired by the QCNN seed."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(540, 120)  # Adjusted for 32x32 RGB images
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class UnifiedQCNNHybrid(nn.Module):
    """Full hybrid QCNN model with a quantum expectation head."""
    def __init__(self, circuit=None, shift: float = 0.0):
        super().__init__()
        self.backbone = QCNNBackbone()
        self.head = Hybrid(in_features=1, circuit=circuit, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x).squeeze(-1)
        probs = self.head(features)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["HybridFunction", "Hybrid", "QCNNBackbone", "UnifiedQCNNHybrid"]
