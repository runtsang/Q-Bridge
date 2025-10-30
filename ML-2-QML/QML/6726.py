import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from typing import Iterable

class QuantumCircuit:
    """Parameterised single‑qubit circuit used as a quantum expectation head."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

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
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy().flatten()
        expectation = circuit.run(thetas)
        result = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        for val in inputs.detach().cpu().numpy():
            exp_plus = circuit.run([val + shift])[0]
            exp_minus = circuit.run([val - shift])[0]
            grads.append((exp_plus - exp_minus) / (2 * shift))
        grad_tensor = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grad_tensor * grad_output, None, None

class QuantumHybridHead(nn.Module):
    """Quantum expectation head that wraps the parameterised circuit."""
    def __init__(self, n_qubits: int = 1, shift: float = np.pi / 2, shots: int = 100):
        super().__init__()
        self.circuit = QuantumCircuit(
            n_qubits, qiskit.Aer.get_backend("qasm_simulator"), shots
        )
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class ClassicalHybridHead(nn.Module):
    """Fallback classical head for compatibility with the ML module."""
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return torch.sigmoid(logits + self.shift)

class UnifiedHybridClassifier(nn.Module):
    """
    Hybrid classifier that can operate in a purely classical mode
    or in a quantum‑augmented mode.
    """
    def __init__(self, mode: str = "quantum", **kwargs):
        super().__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        if mode == "quantum":
            self.head = QuantumHybridHead(
                n_qubits=kwargs.get("n_qubits", 1),
                shift=kwargs.get("shift", np.pi / 2),
                shots=kwargs.get("shots", 100),
            )
        else:
            self.head = ClassicalHybridHead(shift=kwargs.get("shift", 0.0))

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
        probs = self.head(x).unsqueeze(-1)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumCircuit", "HybridFunction", "QuantumHybridHead",
           "ClassicalHybridHead", "UnifiedHybridClassifier"]
