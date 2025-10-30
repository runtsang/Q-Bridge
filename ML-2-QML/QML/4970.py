import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumCircuitWrapper:
    """
    Parameterized circuit that accepts a vector of feature angles.
    Includes an encoding layer, a depth‑controlled variational block,
    and a random entangling layer for added expressivity.
    """
    def __init__(self, n_qubits: int, depth: int, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

        # Parameter vectors
        self.encoding = ParameterVector("x", n_qubits)
        self.weights = ParameterVector("theta", n_qubits * depth)

        # Build circuit
        self.circuit = QuantumCircuit(n_qubits)
        for i, p in enumerate(self.encoding):
            self.circuit.rx(p, i)
        idx = 0
        for _ in range(depth):
            for i in range(n_qubits):
                self.circuit.ry(self.weights[idx], i)
                idx += 1
            for i in range(n_qubits - 1):
                self.circuit.cz(i, i + 1)
        # Add random entanglement
        self.circuit += random_circuit(n_qubits, 2)
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> float:
        """Return the average Z‑expectation across qubits."""
        compiled = transpile(self.circuit, self.backend)
        bind = {p: v for p, v in zip(self.encoding, params)}
        qobj = assemble(compiled, shots=self.shots, parameter_binds=[bind])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        counts = np.array(list(result.values()))
        states = np.array([int(k, 2) for k in result.keys()])
        probs = counts / self.shots
        return np.sum(states * probs) / self.n_qubits

class HybridFunction(torch.autograd.Function):
    """Autograd wrapper that forwards through QuantumCircuitWrapper."""
    @staticmethod
    def forward(ctx, inputs, circuit, shift):
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.numpy())
        out = torch.tensor([expectation], dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        grads = []
        for v, s in zip(inputs.numpy(), shift):
            r = ctx.circuit.run([v + s])
            l = ctx.circuit.run([v - s])
            grads.append(r - l)
        grad = torch.tensor(grads, dtype=torch.float32).unsqueeze(0)
        return grad * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum expectation head that can be plugged into a classical net."""
    def __init__(self, n_qubits: int, depth: int, shift: float = 0.0):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, depth)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.shift)

class QCNet(nn.Module):
    """
    End‑to‑end hybrid classifier mirroring the classical pipeline.
    Convolutional backbone + fully‑connected layers + quantum hybrid head.
    """
    def __init__(self, n_qubits: int = 4, depth: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 1)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(n_qubits, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)
        return self.hybrid(x)

__all__ = ["QuantumCircuitWrapper", "HybridFunction", "Hybrid", "QCNet"]
