import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pennylane as qml

class ChannelAttention(nn.Module):
    """Channel‑wise attention module used before flattening."""
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class QuantumCircuit:
    """Two‑qubit variational circuit executed on a Pennylane device."""
    def __init__(self, n_qubits: int = 2, shots: int = 100, dev_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits, shots=shots)
        self.qnode = self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(theta):
            for i in range(self.n_qubits):
                qml.RY(theta[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        return circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Run the circuit for a batch of theta vectors."""
        results = []
        for theta in thetas:
            val = self.qnode(torch.tensor(theta, dtype=torch.float32))
            results.append(val.item())
        return np.array(results)

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        logits_np = logits.detach().cpu().numpy()
        thetas = np.repeat(logits_np[:, np.newaxis], circuit.n_qubits, axis=1)
        expectation = circuit.run(thetas)
        exp_tensor = torch.tensor(expectation, dtype=torch.float32, device=logits.device)
        prob = torch.sigmoid(exp_tensor)
        ctx.save_for_backward(logits, prob)
        return prob

    @staticmethod
    def backward(ctx, grad_output):
        logits, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        logits_np = logits.detach().cpu().numpy()
        thetas = np.repeat(logits_np[:, np.newaxis], circuit.n_qubits, axis=1)
        grad = []
        for theta in thetas:
            theta_plus = theta + shift
            theta_minus = theta - shift
            exp_plus = circuit.run(theta_plus[np.newaxis, :])[0]
            exp_minus = circuit.run(theta_minus[np.newaxis, :])[0]
            grad.append(exp_plus - exp_minus)
        grad = np.array(grad)
        grad_tensor = torch.tensor(grad, dtype=torch.float32, device=logits.device)
        return grad_tensor * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a two‑qubit circuit."""
    def __init__(self, n_qubits: int = 2, shift: float = np.pi / 2, shots: int = 100):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits=n_qubits, shots=shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.squeeze(-1)
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)

class QCNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.attention = ChannelAttention(15)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            x = self.conv1(dummy)
            x = self.pool(x)
            x = self.drop1(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = self.drop1(x)
            x = self.attention(x)
            x = torch.flatten(x, 1)
            self.flatten_size = x.shape[1]
        self.fc1 = nn.Linear(self.flatten_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(n_qubits=2, shift=np.pi / 2, shots=100)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = self.attention(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)
        probs = self.hybrid(x)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "QCNet", "ChannelAttention"]
