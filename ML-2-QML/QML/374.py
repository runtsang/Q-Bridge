from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumCircuit:
    """Parameterized ansatz with controllable depth for two‑qubit circuits."""
    def __init__(self, n_qubits: int, depth: int = 1, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or AerSimulator()
        # Build a list of parameters: one per qubit per depth layer
        self.params = [qiskit.circuit.Parameter(f"theta_{d}_{q}") for d in range(depth) for q in range(n_qubits)]
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        for d in range(self.depth):
            for qn in range(self.n_qubits):
                qc.ry(self.params[d * self.n_qubits + qn], qn)
            # Entangle neighbouring qubits
            for qn in range(self.n_qubits - 1):
                qc.cx(qn, qn + 1)
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of parameter vectors.

        Parameters
        ----------
        thetas : np.ndarray
            Shape (batch, n_params) or (n_params,).
        """
        if isinstance(thetas, (list, tuple)):
            thetas = np.array(thetas)
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)

        expectations = []
        for row in thetas:
            compiled = transpile(self.circuit, self.backend)
            bind = {p: val for p, val in zip(self.params, row)}
            qobj = assemble(compiled, shots=self.shots, parameter_binds=[bind])
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            # Expectation of Z on the first qubit
            z = np.array([1 if (s & 1) == 0 else -1 for s in states])
            expectations.append(np.sum(z * probs))
        return np.array(expectations)

class HybridFunction(torch.autograd.Function):
    """Autograd wrapper that forwards a batch of parameters to the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.shift = shift
        expectations = circuit.run(inputs.detach().cpu().numpy())
        out = torch.tensor(expectations, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit

        grads = []
        for i in range(inputs.shape[1]):
            shift_vec = np.zeros_like(inputs.cpu().numpy())
            shift_vec[:, i] = shift
            exp_plus = circuit.run(inputs.cpu().numpy() + shift_vec)
            exp_minus = circuit.run(inputs.cpu().numpy() - shift_vec)
            grads.append((exp_plus - exp_minus) / 2.0)
        grads = np.stack(grads, axis=1)
        grads = torch.tensor(grads, device=inputs.device, dtype=inputs.dtype)
        return grads * grad_output.unsqueeze(1), None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through the parameterised circuit."""
    def __init__(self, n_qubits: int, depth: int = 1, backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, depth, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.shift)

class QCNet(nn.Module):
    """Convolutional network with a quantum expectation head."""
    def __init__(self, n_qubits: int = 2, depth: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.n_params = n_qubits * depth
        self.fc3 = nn.Linear(84, self.n_params)
        self.hybrid = Hybrid(n_qubits, depth, shots=1024, shift=np.pi / 2)

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
        x = self.hybrid(x)
        x = x.unsqueeze(-1)
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "QCNet"]
