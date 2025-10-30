import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter

class QuantumCircuitWrapper:
    """Parametrized two‑qubit circuit with expectation‑value readout."""
    def __init__(self, n_qubits: int, backend: qiskit.providers.basebackend.BaseBackend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = Parameter("theta")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        if isinstance(counts, list):
            return np.array([self._expectation(c) for c in counts])
        return np.array([self._expectation(counts)])

    def _expectation(self, count_dict: dict) -> float:
        probs = np.array(list(count_dict.values())) / self.shots
        states = np.array([int(k, 2) for k in count_dict.keys()])
        return np.sum(states * probs)

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.inputs = inputs.detach().cpu().numpy()
        exp_vals = circuit.run(ctx.inputs)
        return torch.tensor(exp_vals, device=inputs.device, dtype=inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs = ctx.inputs
        shift = ctx.shift
        circuit = ctx.circuit
        grad_inputs = torch.zeros_like(inputs)
        for idx, val in enumerate(inputs):
            right = circuit.run([val + shift])[0]
            left = circuit.run([val - shift])[0]
            grad_inputs[idx] = (right - left) / (2 * shift)
        return grad_inputs * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend: qiskit.providers.basebackend.BaseBackend, shots: int, shift: float):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class QuantumHybridClassifier(nn.Module):
    """Hybrid quantum‑classical classifier with a convolutional backbone."""
    def __init__(self, num_classes: int = 2, n_qubits: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)
        backend = AerSimulator()
        self.hybrid = Hybrid(n_qubits, backend, shots=200, shift=np.pi / 4)
        self.classifier = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        quantum_out = self.hybrid(x.squeeze(-1))
        logits = self.classifier(quantum_out)
        probs = torch.softmax(logits, dim=-1)
        return probs

__all__ = ["QuantumHybridClassifier", "Hybrid", "HybridFunction", "QuantumCircuitWrapper"]
