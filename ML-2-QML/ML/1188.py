import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from quantum_module import run_quantum_circuit

class HybridFunction(torch.autograd.Function):
    """Custom autograd function that forwards a linear output to a sigmoid
    activation. The backward pass uses the quantum circuit to compute gradients
    via parameter shift rule."""
    @staticmethod
    def forward(ctx, input_tensor, shift=0.0):
        ctx.save_for_backward(input_tensor)
        ctx.shift = shift
        inputs_np = input_tensor.detach().cpu().numpy().reshape(-1)
        expectations = run_quantum_circuit(inputs_np)
        output = torch.tensor(expectations, dtype=input_tensor.dtype, device=input_tensor.device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        shift = ctx.shift
        inputs_np = input_tensor.detach().cpu().numpy().reshape(-1)
        grad_inputs = []
        for val in inputs_np:
            right = run_quantum_circuit(np.array([val + shift]))
            left = run_quantum_circuit(np.array([val - shift]))
            grad_inputs.append((right - left) / 2.0)
        grad_inputs = torch.tensor(grad_inputs, dtype=input_tensor.dtype, device=input_tensor.device)
        return grad_inputs * grad_output, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, shift: float = np.pi / 2):
        super().__init__()
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.shift)

class Attention1D(nn.Module):
    """Simple 1D attention mechanism over the feature vector."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.attention = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_weights = torch.softmax(self.attention(x), dim=1)
        return x * attn_weights

class HybridQuantumCNN(nn.Module):
    """CNN backbone with attention and a quantum hybrid head."""
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Attention
        self.attention = Attention1D(55815)

        # Hybrid quantum layer
        self.hybrid = Hybrid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)
        x = self.attention(x)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = self.hybrid(x)

        probs = torch.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridFunction", "Hybrid", "Attention1D", "HybridQuantumCNN"]
