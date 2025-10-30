import qiskit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class NoiseLayer(nn.Module):
    """
    Adds Gaussian noise to the activations before the final classifier.
    """
    def __init__(self, noise_std: float = 0.0):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std > 0:
            return x + torch.randn_like(x) * self.noise_std
        return x


class VariationalCircuit:
    """
    A 3‑qubit variational circuit that uses Ry and CX gates.
    """
    def __init__(self, n_qubits: int = 3, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")

        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.ry(self.theta, i)
            if i < self.n_qubits - 1:
                self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> float:
        compiled = transpile(self.circuit, self.backend)
        job = self.backend.run(assemble(compiled, shots=self.shots))
        result = job.result().get_counts()
        return self._expectation(result)

    def _expectation(self, counts: dict) -> float:
        # Expectation value of Z on qubit 0
        exp = 0.0
        for bitstring, count in counts.items():
            z = 1 if bitstring[0] == '0' else -1
            exp += z * count
        return exp / self.shots


class HybridFunction(torch.autograd.Function):
    """
    Differentiable interface between PyTorch and the variational circuit.
    Gradient is estimated via central finite‑difference.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit

        values = inputs.detach().cpu().numpy()
        exp = np.array([circuit.run([v]) for v in values])
        result = torch.tensor(exp, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit

        values = inputs.detach().cpu().numpy()
        grad = []
        for v in values:
            e_plus = circuit.run([v + shift])
            e_minus = circuit.run([v - shift])
            grad.append((e_plus - e_minus) / (2 * shift))
        grad = torch.tensor(grad, dtype=inputs.dtype, device=inputs.device)
        return grad * grad_output, None, None


class QuantumHybridClassifier(nn.Module):
    """
    Quantum‑enabled version of the hybrid classifier.
    The final linear layer is replaced by a 3‑qubit variational circuit.
    """
    def __init__(self, in_features: int = 55815, noise_std: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(in_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.bn = nn.BatchNorm1d(1)
        self.noise = NoiseLayer(noise_std)

        # Quantum head
        self.quantum = VariationalCircuit(n_qubits=3, shots=1024)
        self.shift = np.pi / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
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
        x = self.bn(x)
        x = self.noise(x)
        # Quantum expectation head
        x = HybridFunction.apply(x.squeeze(), self.quantum, self.shift)
        x = torch.sigmoid(x)
        return torch.cat((x, 1 - x), dim=-1)
