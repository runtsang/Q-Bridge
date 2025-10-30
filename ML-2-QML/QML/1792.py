import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumCircuit:
    def __init__(self, n_qubits: int, shots: int = 1024, device: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.shots = shots
        self.device = qml.device(device, wires=n_qubits, shots=shots)

        @qml.qnode(self.device, interface="torch")
        def circuit(theta):
            qml.Hadamard(wires=range(n_qubits))
            qml.ry(theta, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        thetas = np.asarray(thetas)
        if thetas.ndim == 0:
            thetas = thetas.reshape(1)
        expectations = []
        for theta in thetas:
            exp = self.circuit(torch.tensor(theta, dtype=torch.float32))
            expectations.append(exp.item())
        return np.array(expectations)

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy()
        expectations = circuit.run(thetas)
        result = torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        thetas = inputs.detach().cpu().numpy()
        gradients = []
        for theta in thetas:
            right = ctx.circuit.run(np.array([theta + shift]))[0]
            left = ctx.circuit.run(np.array([theta - shift]))[0]
            grad = (right - left) / 2.0
            gradients.append(grad)
        gradients = torch.tensor(gradients, dtype=inputs.dtype, device=inputs.device)
        return gradients * grad_output, None, None

class Hybrid(nn.Module):
    def __init__(self, n_qubits: int, shots: int, shift: float):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor):
        return HybridFunction.apply(torch.squeeze(inputs), self.circuit, self.shift)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels!= out_channels or stride!= 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class QCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.resblock = ResidualBlock(6, 15, stride=1)
        self.drop2 = nn.Dropout2d(p=0.5)

        dummy = torch.zeros(1, 3, 32, 32)
        dummy_feat = self._forward_features(dummy, return_feat=True)
        flat_features = dummy_feat.shape[1]

        self.fc1 = nn.Linear(flat_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(n_qubits=1, shots=1024, shift=np.pi / 2)

    def _forward_features(self, x, return_feat=False):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = self.resblock(x)
        x = self.pool(x)
        x = self.drop2(x)
        x = torch.flatten(x, 1)
        if return_feat:
            return x
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, inputs: torch.Tensor):
        x = self._forward_features(inputs)
        x = self.hybrid(x)
        probs = torch.sigmoid(x)
        return torch.stack([probs, 1 - probs], dim=1)

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "ResidualBlock", "QCNet"]
