import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter

class QuantumCircuitModule(nn.Module):
    """Two‑qubit variational circuit on Aer."""
    def __init__(self, n_qubits: int = 2, shots: int = 1024):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("aer_simulator")
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
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
        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge to the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitModule, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = circuit.run(inputs.tolist())
        out = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs) * ctx.shift
        grads = []
        for idx, val in enumerate(inputs.tolist()):
            exp_plus = ctx.circuit.run([val + shift[idx]])
            exp_minus = ctx.circuit.run([val - shift[idx]])
            grads.append(exp_plus - exp_minus)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

class HybridLayer(nn.Module):
    """Dense layer that forwards activations through the quantum circuit."""
    def __init__(self, n_qubits: int = 2, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuitModule(n_qubits, shots)
        self.shift = shift
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class QuantumKernelHybridNet(nn.Module):
    """CNN backbone followed by a quantum expectation head."""
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
        self.hybrid = HybridLayer(n_qubits=2, shots=1024, shift=np.pi / 2)
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
        out = self.hybrid(x).T
        return torch.cat((out, 1 - out), dim=-1)

__all__ = ["QuantumKernelHybridNet"]
