import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumCircuit:
    """Parameterised two‑qubit circuit executed on a chosen backend."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        for i in range(n_qubits):
            self.circuit.h(i)
            self.circuit.ry(self.theta, i)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        exp = 0
        for bitstring, count in counts.items():
            exp += int(bitstring[0]) * count
        exp /= self.shots
        return np.array([exp])

class HybridFunction(torch.autograd.Function):
    """Bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy()
        expectations = []
        for theta in thetas:
            expectation = circuit.run(np.array([theta, theta]))[0]
            expectations.append(expectation)
        expectations = torch.tensor(expectations, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, expectations)
        return expectations

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for inp in inputs:
            theta_plus = inp + shift
            theta_minus = inp - shift
            exp_plus = ctx.circuit.run(np.array([theta_plus, theta_plus]))[0]
            exp_minus = ctx.circuit.run(np.array([theta_minus, theta_minus]))[0]
            grad = (exp_plus - exp_minus) / (2 * shift)
            grads.append(grad)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None

class HybridQuantumLayer(nn.Module):
    """Quantum layer that feeds the CNN output into a variational circuit."""
    def __init__(self, n_qubits: int, backend, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x.squeeze(), self.circuit, self.shift)

class HybridQuantumCNN(nn.Module):
    """CNN with a quantum hybrid head and a skip connection."""
    def __init__(self) -> None:
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
        self.hybrid = HybridQuantumLayer(2, backend, shots=200, shift=np.pi / 2)

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
        skip = F.relu(self.fc2(x))
        x = self.fc3(skip)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["HybridQuantumCNN", "HybridQuantumLayer", "HybridFunction", "QuantumCircuit"]
