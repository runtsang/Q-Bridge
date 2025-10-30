import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit, assemble, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter

class BayesianCalibration(nn.Module):
    """Calibrates logits using a prior‑weighted sigmoid."""
    def __init__(self, prior: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.prior = prior
        self.eps = eps

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((logits - self.prior) / (1 + self.prior))

class MultiQubitQuantumCircuit:
    """Parameterized multi‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = [Parameter(f"theta_{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self.circuit.h(i)
        for i, t in enumerate(self.theta):
            self.circuit.ry(t, i)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        param_binds = [{t: val for t, val in zip(self.theta, thetas)}]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        return self._expectation(result)

    def _expectation(self, counts):
        probs = np.zeros(self.n_qubits)
        for bitstring, cnt in counts.items():
            prob = cnt / self.shots
            for i, bit in enumerate(reversed(bitstring)):
                probs[i] += prob * (1 if bit == '0' else -1)
        return probs

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: MultiQubitQuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.squeeze().tolist()
        exp_vals = ctx.circuit.run(thetas)
        logit = torch.tensor([np.sum(exp_vals)], dtype=torch.float32)
        ctx.save_for_backward(inputs)
        return logit

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.squeeze().tolist():
            exp_r = ctx.circuit.run([val + shift])
            exp_l = ctx.circuit.run([val - shift])
            grads.append(exp_r - exp_l)
        grad = torch.tensor([sum(grads)], dtype=torch.float32)
        return grad * grad_output, None, None

class HybridLayer(nn.Module):
    """Hybrid layer that forwards activations through a multi‑qubit quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = MultiQubitQuantumCircuit(n_qubits, backend, shots)
        self.shift = shift
        self.linear = nn.Linear(1, n_qubits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Map scalar input to qubit angles
        angles = self.linear(inputs.unsqueeze(-1))  # (batch, 1, n_qubits)
        angles = angles.squeeze(-1)  # (batch, n_qubits)
        outputs = []
        for a in angles:
            exp_vals = self.quantum_circuit.run(a.detach().cpu().numpy())
            logit = np.sum(exp_vals)
            outputs.append(logit)
        return torch.tensor(outputs, dtype=torch.float32).unsqueeze(-1)

class QuantumHybridBinaryClassifier(nn.Module):
    """Convolutional network followed by a quantum expectation head."""
    def __init__(self, n_qubits: int = 4, shift: float = 0.5, shots: int = 100) -> None:
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
        self.hybrid = HybridLayer(n_qubits, backend, shots, shift)
        self.calib = BayesianCalibration()

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
        probs = self.hybrid(x)
        probs = self.calib(probs)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["MultiQubitQuantumCircuit", "HybridFunction", "HybridLayer", "QuantumHybridBinaryClassifier"]
