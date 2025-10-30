"""Enhanced hybrid quantum‑classical binary classifier with a 4‑qubit variational circuit.

The module defines QCNet, a convolutional neural network followed by a
parameterised quantum expectation head.  The quantum circuit now contains
two layers of RX‑RZ rotations and CX entanglement, and the gradient is
computed using the parameter‑shift rule for numerical stability.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit as QC, transpile, assemble
from qiskit.providers.aer import AerSimulator

class VariationalQuantumCircuit:
    """4‑qubit variational circuit with two rotation layers and CX entanglement."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = QC(n_qubits)
        # Parameterised rotation layer
        self.theta = QC.circuit.Parameter("theta")
        for q in range(n_qubits):
            self.circuit.rx(self.theta, q)
            self.circuit.rz(self.theta, q)
        # Entanglement
        for q in range(n_qubits - 1):
            self.circuit.cx(q, q + 1)
        self.circuit.barrier()
        # Second rotation layer
        for q in range(n_qubits):
            self.circuit.rx(self.theta, q)
            self.circuit.rz(self.theta, q)
        # Measurement
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit for the given parameters."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: p} for p in params],
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
    """Differentiable interface between PyTorch and the variational circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalQuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        # Convert to numpy
        params = inputs.detach().cpu().numpy()
        exp_vals = circuit.run(params)
        out = torch.tensor(exp_vals, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        params = inputs.detach().cpu().numpy()
        grads = []
        for i, val in enumerate(params):
            right = ctx.circuit.run([val + shift])
            left = ctx.circuit.run([val - shift])
            grads.append((right - left) / 2)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through the variational circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.circuit = VariationalQuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class QCNet(nn.Module):
    """Convolutional network followed by a 4‑qubit variational quantum head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(64 * 7 * 7, 256, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1, bias=False)

        backend = AerSimulator()
        self.hybrid = Hybrid(n_qubits=4, backend=backend, shots=1024, shift=np.pi/2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x).squeeze(-1)
        probs = self.hybrid(x)
        probs = torch.sigmoid(probs)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["QCNet", "Hybrid", "HybridFunction", "VariationalQuantumCircuit"]
