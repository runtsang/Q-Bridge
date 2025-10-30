import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile

class QuantumCircuitWrapper:
    """
    Thin wrapper around a single‑qubit parameterized circuit executed on Aer.
    """
    def __init__(self, n_qubits: int, backend, shots: int):
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, qubits)
        self.circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(counts):
            counts_arr = np.array(list(counts.values()))
            states = np.array(list(counts.keys())).astype(float)
            probs = counts_arr / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge between PyTorch and the quantum circuit using the
    parameter‑shift rule for gradient estimation.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.circuit = circuit
        ctx.shift = shift
        expectation = ctx.circuit.run(inputs.tolist())
        ctx.save_for_backward(inputs)
        return torch.tensor(expectation, dtype=inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs:
            right = ctx.circuit.run([val.item() + shift])
            left = ctx.circuit.run([val.item() - shift])
            grads.append((right - left) / (2 * shift))
        grads = torch.tensor(grads, dtype=grad_output.dtype)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """
    Hybrid layer that forwards activations through a quantum circuit.
    """
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)

class EstimatorQNN(nn.Module):
    """
    Hybrid neural‑regressor that combines a convolutional backbone with a
    quantum expectation head.
    """
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
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(1, backend, shots=100, shift=np.pi / 2)

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
        x = self.hybrid(x).squeeze()
        return torch.sigmoid(x)

__all__ = ["EstimatorQNN"]
