import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuantumCircuit:
    """Parameterized two‑qubit circuit executed on a PennyLane device."""
    def __init__(self, n_qubits: int, dev: qml.Device, shift: float = np.pi / 2) -> None:
        self.n_qubits = n_qubits
        self.dev = dev
        self.shift = shift
        self.theta = qml.numpy.array(np.zeros(n_qubits))
        self.qnode = qml.QNode(self._circuit, self.dev, interface='torch')

    def _circuit(self, params: np.ndarray) -> float:
        for i in range(self.n_qubits):
            qml.Hadamard(i)
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        # Fully‑connected entanglement pattern
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    def run(self, inputs: np.ndarray) -> torch.Tensor:
        """
        Execute the circuit for a batch of input angles.
        Returns a torch tensor of expectation values.
        """
        # Broadcast inputs if necessary
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        outputs = self.qnode(inputs)
        return outputs


class HybridFunction(torch.autograd.Function):
    """Autograd wrapper that evaluates the quantum circuit and its gradients."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit) -> torch.Tensor:
        ctx.circuit = circuit
        # Ensure inputs are detached for gradient calculation
        outputs = circuit.run(inputs.detach().cpu().numpy())
        return torch.tensor(outputs, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        circuit = ctx.circuit
        # Parameter‑shift rule
        def circuit_func(params):
            return circuit.qnode(params)

        grads = qml.gradients.param_shift(circuit_func)(circuit.theta)
        grads = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device)
        return grads * grad_output, None


class Hybrid(nn.Module):
    """Layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int = 2, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(
            n_qubits=n_qubits,
            dev=qml.device("default.qubit", wires=n_qubits),
            shift=shift
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit)


class QCNet(nn.Module):
    """
    Residual CNN followed by a quantum expectation head.
    The architecture mirrors the extended classical counterpart.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.res1 = ResidualBlock(6, 6)
        self.res2 = ResidualBlock(6, 6)

        self.fc1 = nn.Linear(294, 120, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(120)
        self.drop_fc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.bn_fc2 = nn.BatchNorm1d(84)
        self.drop_fc2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(84, 1, bias=False)

        self.hybrid = Hybrid(n_qubits=2, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = self.pool(x)
        x = self.drop1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.drop_fc1(x)
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.drop_fc2(x)
        x = self.fc3(x)
        prob = self.hybrid(x)
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "QCNet", "ResidualBlock"]
