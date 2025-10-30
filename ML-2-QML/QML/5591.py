import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import Aer, transpile, assemble

class QuantumCircuit:
    """Parameterized two‑qubit ansatz used as the quantum head."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        # simple entangling ansatz
        self.circuit.h(range(n_qubits))
        for q in range(n_qubits):
            self.circuit.ry(self.theta, q)
        for q in range(n_qubits - 1):
            self.circuit.cz(q, q + 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        param_binds = [{self.theta: t} for t in thetas]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
            return np.sum(states * probs)
        return np.array([expectation(counts)])

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # expectation value for each element in the batch
        thetas = inputs.detach().cpu().numpy().flatten()
        exp_vals = circuit.run(thetas)
        result = torch.tensor(exp_vals, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.cpu().numpy():
            right = ctx.circuit.run([val + shift])[0]
            left  = ctx.circuit.run([val - shift])[0]
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through the quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # flatten to one‑dimensional batch
        flat = inputs.view(-1)
        return HybridFunction.apply(flat, self.circuit, self.shift).unsqueeze(1)

class HybridBinaryClassifier(nn.Module):
    """
    Hybrid CNN followed by a variational quantum expectation head.
    Matches the public API of the classical counterpart.
    """
    def __init__(self, n_qubits: int = 2, shots: int = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6 * 15 * 15, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits, backend, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = self.hybrid(logits).squeeze(-1)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridBinaryClassifier"]
