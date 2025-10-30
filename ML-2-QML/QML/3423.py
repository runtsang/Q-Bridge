"""Hybrid classical‑quantum binary classifier – quantum implementation.

The architecture mirrors the classical version but replaces the
quanvolution filter and the classification head with genuine quantum
operations.  The network demonstrates the synergy between CNN feature
extraction and variational quantum circuits.

Key features from the seed projects:

* Quantum circuit wrapper (QuantumCircuit) – from the original QML seed.
* HybridFunction that passes gradients via finite‑difference – from the
  original QML seed.
* QuanvCircuit – quantum filter – from Conv.py.
* Standard CNN backbone identical to the classical version.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import assemble, transpile, Aer, execute
from qiskit.circuit.random import random_circuit

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Quantum circuit for the final classification head -----
class QuantumCircuit:
    """Parameterized two‑qubit circuit executed on a simulator."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
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
        # compute expectation value of Z on the first qubit
        def expectation(count_dict):
            total = 0.0
            for state, count in count_dict.items():
                # state string e.g. '01'
                z = 1 if state[0] == "0" else -1  # |0> -> +1, |1> -> -1
                total += z * count
            return total / self.shots
        return np.array([expectation(result)])

# ----- Hybrid interface between PyTorch and quantum circuit -----
class HybridFunction(torch.autograd.Function):
    """Forward pass through a quantum circuit; backward via finite‑difference."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift
        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)
        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

class Hybrid(nn.Module):
    """Quantum hybrid layer performing a variational expectation."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

# ----- Quantum quanvolution filter -----
class QuantumQuanvFilter(nn.Module):
    """Quantum filter that emulates the classical quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 127, backend=None, shots: int = 100):
        super().__init__()
        if backend is None:
            backend = Aer.get_backend("qasm_simulator")
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.backend = backend
        self.shots = shots
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Run the quantum circuit on a single 2‑D patch."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)
        # compute average probability of measuring |1> across all qubits
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the quantum filter to each patch of the feature map.
        x : (batch, channels, h, w)
        Returns: (batch, channels)
        """
        batch, channels, h, w = x.shape
        # extract patches using unfold
        patches = F.unfold(x, kernel_size=self.kernel_size)  # (batch, ks*ks*channels, L)
        patches = patches.view(batch, channels, self.kernel_size, self.kernel_size, -1)
        outputs = []
        for b in range(batch):
            channel_out = []
            for c in range(channels):
                channel_patches = patches[b, c].permute(1, 0).cpu().numpy()  # (L, ks, ks)
                activations = np.array([self.run(p) for p in channel_patches])
                channel_out.append(activations.mean())
            outputs.append(channel_out)
        return torch.tensor(outputs, device=x.device, dtype=torch.float32)

# ----- Hybrid network with quantum filter and head -----
class HybridQCNet(nn.Module):
    """Quantum‑enhanced CNN classifier."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(15, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = Aer.get_backend("qasm_simulator")
        self.quantum_filter = QuantumQuanvFilter(kernel_size=2, threshold=127, backend=backend, shots=100)
        self.hybrid = Hybrid(self.fc3.out_features, backend, shots=100, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        # apply quantum filter
        x = self.quantum_filter(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.hybrid(x).T
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["HybridQCNet"]
