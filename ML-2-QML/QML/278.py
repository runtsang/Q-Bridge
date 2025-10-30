import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class QuantumCircuit:
    """Parameterized 2-qubit circuit with a simple feature map."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        # Simple feature map: H and RY(theta) on each qubit
        for q in range(n_qubits):
            self.circuit.h(q)
            self.circuit.ry(self.theta, q)
        self.circuit.measure_all()

    def run(self, theta_values: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: val} for val in theta_values],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(counts):
            total = sum(counts.values())
            exp = 0.0
            for state, count in counts.items():
                # Use first qubit for Z expectation
                z = 1.0 if state[0] == '0' else -1.0
                exp += z * (count / total)
            return exp

        if isinstance(result, list):
            return np.array([expectation(c) for c in result])
        else:
            return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Autograd wrapper for the quantum expectation."""
    @staticmethod
    def forward(ctx, logits: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        theta_vals = logits.detach().cpu().numpy()
        expectations = ctx.circuit.run(theta_vals)
        out = torch.tensor(expectations, device=logits.device, dtype=logits.dtype)
        ctx.save_for_backward(logits, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = []
        for val in logits.detach().cpu().numpy():
            grad = (ctx.circuit.run(np.array([val + shift])) -
                    ctx.circuit.run(np.array([val - shift]))) / 2.0
            grad_inputs.append(grad)
        grad_tensor = torch.tensor(grad_inputs, device=logits.device, dtype=logits.dtype)
        return grad_tensor * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid head that forwards through the quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 1:
            x = torch.squeeze(x)
        return HybridFunction.apply(x, self.quantum_circuit, self.shift)

class HybridQuantumBinaryClassifier(nn.Module):
    """
    CNN backbone followed by a quantum expectation head.
    """
    def __init__(self, n_qubits: int = 2, shots: int = 200, shift: float = np.pi / 2):
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
        self.hybrid = Hybrid(n_qubits, backend, shots, shift)

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
        # Quantum head
        x = self.hybrid(x).squeeze()
        probs = torch.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQuantumBinaryClassifier"]
