import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import Aer, transpile, assemble
from qiskit.circuit import Parameter

class ResidualBlock(nn.Module):
    """Adds a residual connection around two linear layers."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin1 = nn.Linear(in_features, out_features)
        self.lin2 = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.lin1(x))
        out = F.relu(self.lin2(out))
        return out + residual

class QuantumCircuit2Qubit:
    """Two‑qubit variational circuit with a single‑parameter ansatz."""
    def __init__(self, backend: qiskit.providers.BaseBackend, shots: int):
        self.backend = backend
        self.shots = shots
        self.theta0 = Parameter('theta0')
        self.theta1 = Parameter('theta1')
        self.qc = qiskit.QuantumCircuit(2)
        self.qc.h(0)
        self.qc.h(1)
        self.qc.ry(self.theta0, 0)
        self.qc.ry(self.theta1, 1)
        self.qc.cx(0, 1)
        self.qc.measure_all()

    def run(self, angles: np.ndarray) -> np.ndarray:
        """Execute the circuit for the given pair of angles and return a 2‑dimensional probability vector."""
        bound_qc = self.qc.bind_parameters({self.theta0: angles[0], self.theta1: angles[1]})
        compiled = transpile(bound_qc, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        # Compute probability that each qubit is measured in state |1⟩.
        p0_1 = (counts.get('10', 0) + counts.get('11', 0)) / self.shots
        p1_1 = (counts.get('01', 0) + counts.get('11', 0)) / self.shots
        return np.array([p0_1, p1_1])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit2Qubit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        probs = []
        for angles in inputs:
            probs.append(circuit.run(angles.tolist()))
        result = torch.tensor(probs, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        for angles in inputs:
            grad_angles = []
            for i in range(len(angles)):
                ang_plus = angles.clone()
                ang_minus = angles.clone()
                ang_plus[i] += shift
                ang_minus[i] -= shift
                p_plus = circuit.run(ang_plus.tolist())
                p_minus = circuit.run(ang_minus.tolist())
                grad_angles.append(p_plus - p_minus)
            grads.append(torch.tensor(grad_angles))
        grads = torch.stack(grads)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a two‑qubit quantum circuit."""
    def __init__(self, backend, shots: int, shift: float):
        super().__init__()
        self.circuit = QuantumCircuit2Qubit(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class HybridClassifier(nn.Module):
    """CNN followed by a quantum expectation head that outputs a 2‑dimensional probability vector."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.residual = ResidualBlock(120, 120)
        self.fc3 = nn.Linear(84, 2)
        backend = Aer.get_backend('aer_simulator')
        self.hybrid = Hybrid(backend, shots=200, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.residual(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return probs
