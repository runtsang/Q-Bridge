"""QuantumHybridClassifier – quantum hybrid head using Qiskit."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class QuantumCircuit:
    """
    Parameterised two‑qubit circuit that returns the expectation
    value of Pauli‑Z on the first qubit.
    """
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots

        self.circuit = qiskit.QuantumCircuit(n_qubits)
        qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        # Simple entangling block
        self.circuit.h(qubits)
        self.circuit.ry(self.theta, qubits)
        self.circuit.cx(qubits[0], qubits[1])
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of parameter values.

        Parameters
        ----------
        thetas : np.ndarray
            1‑D array of theta values (shape (batch,)).
        """
        compiled = transpile(self.circuit, self.backend)
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
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """
    Autograd wrapper that forwards a scalar through a quantum circuit
    and returns the expectation value as a torch tensor.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # inputs is (batch,) tensor
        thetas = inputs.detach().cpu().numpy()
        exp_vals = circuit.run(thetas)
        result = torch.tensor(exp_vals, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        thetas = inputs.detach().cpu().numpy()
        grads = []
        for theta in thetas:
            # Parameter‑shift rule
            right = ctx.circuit.run(np.array([theta + shift]))
            left = ctx.circuit.run(np.array([theta - shift]))
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=inputs.dtype, device=inputs.device)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """
    Hybrid head that maps a scalar feature to a probability using a
    quantum expectation value.
    """
    def __init__(self, n_qubits: int = 2, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend=None, shots=shots)
        self.shift = shift
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape (batch, 1)
        flat = inputs.view(-1)
        exp_vals = HybridFunction.apply(flat, self.circuit, self.shift)
        probs = self.sigmoid(exp_vals).view(-1, 1)
        return probs

class QuantumHybridClassifier(nn.Module):
    """
    Convolutional backbone followed by a quantum hybrid head.
    Mirrors the structure of the original hybrid network.
    """
    def __init__(self, n_qubits: int = 2, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(n_qubits, shots=shots, shift=shift)

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
        probs = self.hybrid(x)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["QuantumHybridClassifier"]
