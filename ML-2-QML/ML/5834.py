import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class QuantumCircuit:
    """Parameterised two‑qubit circuit executed on Qiskit Aer."""
    def __init__(self, backend: qiskit.providers.BaseBackend, shots: int = 1024):
        self.backend = backend
        self.shots = shots
        self._circuit = qiskit.QuantumCircuit(2)
        theta = qiskit.circuit.Parameter("theta")
        self.theta = theta
        self._circuit.h([0, 1])
        self._circuit.barrier()
        self._circuit.ry(theta, 0)
        self._circuit.rz(theta, 1)
        self._circuit.cx(0, 1)
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            probs = np.array(list(count_dict.values())) / self.shots
            states = np.array([int(k, 2) for k in count_dict.keys()])
            return np.sum(states * probs)
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit with residual."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit1: QuantumCircuit, circuit2: QuantumCircuit,
                shift: float, mode: str = "normal") -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit1 = circuit1
        ctx.circuit2 = circuit2
        # ensemble of two circuits
        vals1 = ctx.circuit1.run(inputs.tolist())
        vals2 = ctx.circuit2.run(inputs.tolist())
        mean = (vals1 + vals2) / 2.0
        var = ((vals1 - mean)**2 + (vals2 - mean)**2) / 2.0
        # residual: add input to quantum output
        result = mean + inputs
        ctx.save_for_backward(inputs, torch.tensor(mean), torch.tensor(var))
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, mean, var = ctx.saved_tensors
        grad_inputs = grad_output  # derivative of residual part is 1
        return grad_inputs, None, None, None, None

class Hybrid(nn.Module):
    """Hybrid head that runs two quantum circuits and returns a probability."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float = np.pi / 2):
        super().__init__()
        self.circuit1 = QuantumCircuit(backend, shots)
        self.circuit2 = QuantumCircuit(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(), self.circuit1, self.circuit2, self.shift)

class QCNet(nn.Module):
    """Convolutional network with a quantum hybrid head."""
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
        backend = AerSimulator()
        self.hybrid = Hybrid(2, backend, shots=500, shift=np.pi / 2)

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

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "QCNet"]
