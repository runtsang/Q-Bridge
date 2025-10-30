"""Hybrid attention‑based binary classifier with a quantum self‑attention block
and a quantum‑style expectation head.

The network follows the same architecture as the classical version but replaces
the self‑attention and the head with their quantum counterparts.  The
self‑attention block is implemented as a two‑qubit circuit that applies
parameterised rotations and controlled‑X entanglement, and the head is a
parameterised two‑qubit expectation circuit.  Both components expose a
differentiable PyTorch interface via custom autograd functions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# --- Quantum self‑attention ----------------------------------------------------
class QuantumSelfAttention:
    """Basic quantum circuit representing a self‑attention style block."""

    def __init__(self, n_qubits: int, backend):
        self.n_qubits = n_qubits
        self.backend = backend
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        counts = job.result().get_counts(circuit)
        return self._counts_to_expectation(counts)

    @staticmethod
    def _counts_to_expectation(counts: dict, n_qubits: int = 4) -> np.ndarray:
        """Convert measurement counts to a vector of qubit‑wise expectation values."""
        expectations = np.zeros(n_qubits)
        total = sum(counts.values())
        for bitstring, cnt in counts.items():
            prob = cnt / total
            for i, bit in enumerate(reversed(bitstring)):
                expectations[i] += ((-1) ** int(bit)) * prob
        return expectations


# --- Quantum circuit for hybrid head -----------------------------------------
class QuantumCircuitWrapper:
    """Wrapper around a parametrised two‑qubit circuit executed on Aer."""

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

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
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
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)


# --- Main network -------------------------------------------------------------
class HybridAttentionQCNet(nn.Module):
    """Convolutional binary classifier with a quantum self‑attention block
    and a quantum hybrid head that mimics a quantum expectation layer.
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected head
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum self‑attention parameters
        self.attn_rotation = nn.Parameter(torch.randn(4 * 3))
        self.attn_entangle = nn.Parameter(torch.randn(4 - 1))
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.quantum_self_attention = QuantumSelfAttention(n_qubits=4, backend=backend)

        # Hybrid head
        self.hybrid = Hybrid(n_qubits=1, backend=backend, shots=100, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Convolutional feature extraction
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # Quantum self‑attention on flattened features
        rotation_np = self.attn_rotation.detach().cpu().numpy()
        entangle_np = self.attn_entangle.detach().cpu().numpy()
        attn_out = self.quantum_self_attention.run(rotation_np, entangle_np, shots=1024)
        attn_out = torch.from_numpy(attn_out).to(inputs.device)

        # Feed‑forward layers
        x = F.relu(self.fc1(attn_out))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Hybrid head
        probabilities = self.hybrid(x)
        return torch.cat((probabilities, 1 - probabilities), dim=-1)


__all__ = ["HybridAttentionQCNet"]
