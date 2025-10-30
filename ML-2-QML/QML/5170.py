from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class QuantumCircuit:
    """
    Parameterised ansatz used by the quantum head.  The circuit consists of
    an initial RX data‑encoding layer followed by alternating parameterised
    RY rotations and CZ entanglers.
    """
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots

        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.encoding = ParameterVector("x", n_qubits)
        self.weights = ParameterVector("theta", n_qubits * 2)

        # Data‑encoding
        for q, param in zip(range(n_qubits), self.encoding):
            self.circuit.rx(param, q)

        # Variational layers
        for i in range(2):
            for q in range(n_qubits):
                self.circuit.ry(self.weights[i * n_qubits + q], q)
            for q in range(n_qubits - 1):
                self.circuit.cz(q, q + 1)

        # Measure all qubits
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for the given parameter vector.
        The vector consists of the encoded data followed by the variational
        angles.  The method returns an array of expectation values of
        the Pauli‑Z observable on each qubit.
        """
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.encoding[i]: params[i] for i in range(self.n_qubits)},
                             {self.weights[i]: params[self.n_qubits + i] for i in range(self.n_qubits * 2)}],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict: dict[str, int]) -> np.ndarray:
            total = sum(count_dict.values())
            exp = np.zeros(self.n_qubits)
            for bitstring, count in count_dict.items():
                bits = np.array([int(b) for b in bitstring[::-1]])
                probs = count / total
                exp += probs * (1 - 2 * bits)  # Z eigenvalues: +1 for 0, -1 for 1
            return exp

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return expectation(result)

class HybridFunction(torch.autograd.Function):
    """
    Bridges PyTorch and the quantum circuit.  The forward pass evaluates
    the circuit and returns the expectation values.  The backward pass
    implements the parameter‑shift rule for all variational angles.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Inputs are flattened: first n_qubits are encoded, rest are variational
        params = inputs.detach().cpu().numpy()
        expectations = circuit.run(params)
        out = torch.tensor(expectations, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.cpu().numpy()) * ctx.shift
        grads = []

        for idx in range(inputs.numel()):
            # Parameter‑shift rule
            right = ctx.circuit.run((inputs.cpu().numpy() + shift)[idx])
            left = ctx.circuit.run((inputs.cpu().numpy() - shift)[idx])
            grads.append(right - left)

        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None


class Hybrid(nn.Module):
    """
    Quantum head that maps a vector of parameters to a vector of
    expectation values.  The head is parameterised by the number of
    qubits and a backend/shots configuration.
    """
    def __init__(self, n_qubits: int, backend, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Flatten inputs to match the circuit's expectation
        flat = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(flat, self.circuit, self.shift)


class HybridClassifier(nn.Module):
    """
    Hybrid quantum‑classical classifier that mirrors the classical
    `HybridClassifier`.  It uses a small convolutional backbone to
    extract features, then projects the features to a vector that
    feeds the variational circuit.  The output of the circuit is summed,
    passed through a sigmoid head, and returned as a probability
    distribution over two classes.
    """
    def __init__(self, n_qubits: int = 4, shift: float = np.pi / 2) -> None:
        super().__init__()
        # Simple CNN backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.flatten = nn.Flatten()

        # Project to a vector that matches the circuit input size
        # (n_qubits for encoding + 2*n_qubits for variational params)
        proj_dim = n_qubits * 3
        self.proj = nn.Linear(55815, proj_dim)

        # Quantum head
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits, backend, shots=512, shift=shift)

        # Classical head to produce a single logit
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = self.flatten(x)
        x = self.proj(x)

        # Run quantum circuit
        q_expect = self.hybrid(x)

        # Sum expectations to form a logit
        logits = self.head(q_expect)

        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "HybridClassifier"]
