"""ConvHybridNet: quantum implementation that uses a quantum filter and a quantum expectation head."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile

class QuantumFilter:
    """Quantum filter that mimics a convolutional kernel using a parameterized circuit.

    The filter operates on a 2‑D patch of size `kernel_size` and returns the
    average probability of measuring |1> across all qubits.
    """
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float = 0.5) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Execute the circuit on a single 2‑D patch.

        Parameters
        ----------
        data: np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1>.
        """
        flat = np.reshape(data, (self.n_qubits,))
        param_binds = [{self.theta[i]: np.pi if flat[i] > self.threshold else 0.0
                        for i in range(self.n_qubits)}]
        job = qiskit.execute(self._circuit, self.backend, shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for bitstring, freq in result.items():
            ones = bitstring.count("1")
            counts += ones * freq
        return counts / (self.shots * self.n_qubits)

class QuantumCircuit:
    """Parameterized circuit used as the quantum head."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        full_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(full_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, full_qubits)
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
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

class QuantumHybridFunction(torch.autograd.Function):
    """Differentiable wrapper that executes a quantum circuit for a batch of angles."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.detach().cpu().numpy())
        return torch.tensor([expectation])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shift = np.ones_like(ctx.circuit) * ctx.shift
        grads = []
        for idx, val in enumerate(ctx.circuit.run(inputs)):
            right = ctx.circuit.run(np.array([val + shift[idx]]))
            left = ctx.circuit.run(np.array([val - shift[idx]]))
            grads.append(right - left)
        grads = torch.tensor([grads]).float()
        return grads * grad_output.float(), None, None

class QuantumHybridHead(nn.Module):
    """Hybrid head that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(inputs.squeeze(), self.quantum_circuit, self.shift)

class ConvHybridNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.hybrid = QuantumHybridHead(n_qubits=1, backend=backend, shots=100, shift=np.pi / 2)

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
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumFilter", "QuantumCircuit", "QuantumHybridHead",
           "QuantumHybridFunction", "ConvHybridNet"]
