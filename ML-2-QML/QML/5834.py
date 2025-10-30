import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

# Quantum device with 2 qubits
dev = qml.device("default.qubit", wires=2, shots=1024)

def two_qubit_ansatz(params, wires):
    qml.Hadamard(wires=wires[0])
    qml.Hadamard(wires=wires[1])
    qml.RY(params[0], wires=wires[0])
    qml.RZ(params[0], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])

@qml.qnode(dev, interface="torch", diff_method="finite-diff")
def qnode_single(params):
    two_qubit_ansatz(params, wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface="torch", diff_method="finite-diff")
def qnode_weighted(params):
    two_qubit_ansatz(params, wires=[0, 1])
    return qml.expval(0.5 * qml.PauliZ(0) + 0.5 * qml.PauliZ(1))

class HybridFunction(torch.autograd.Function):
    """Bridge between PyTorch and PennyLane with ensemble variance."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float):
        # ensemble of two circuits
        out1 = qnode_single(inputs)
        out2 = qnode_weighted(inputs)
        mean = (out1 + out2) / 2.0
        var = ((out1 - mean)**2 + (out2 - mean)**2) / 2.0
        result = mean + inputs  # residual
        ctx.save_for_backward(inputs, mean, var)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, mean, var = ctx.saved_tensors
        grad_inputs = grad_output  # residual derivative
        return grad_inputs, None

class Hybrid(nn.Module):
    """Hybrid head using PennyLane circuits."""
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(), self.shift)

class QCNet(nn.Module):
    """Convolutional network with a PennyLane quantum hybrid head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(shift=0.0)

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
        probs = torch.sigmoid(self.hybrid(x))
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridFunction", "Hybrid", "QCNet"]
