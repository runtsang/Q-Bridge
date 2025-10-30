"""Unified ML/QML regression model with residual classical network and Qiskit variational circuit.

This module is intentionally self‑contained: it can be dropped next to the original
QuantumRegression.py and used as a drop‑in replacement.  The class
UnifiedQuantumRegression implements a hybrid architecture that
combines a residual MLP with a 1‑qubit variational circuit whose
expectation values are used as features for the final linear head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import qiskit
from qiskit import transpile, assemble

# --------------------------------------------------------------------------- #
# Data generation and dataset
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int, samples: int, noise: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data from a superposition‑based target.
    """
    rng = np.random.default_rng()
    X = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles) + noise * rng.normal(size=samples)
    return X, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Simple ``torch.utils.data.Dataset`` wrapper around the synthetic data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Classical feature extractor
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    """A single residual block with an optional linear shortcut."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )
        self.shortcut = (
            nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.relu(self.fc(x) + self.shortcut(x))


class ClassicalFeatureExtractor(nn.Module):
    """Stacked residual blocks followed by a ReLU."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(input_dim, 64),
            ResidualBlock(64, 64),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

# --------------------------------------------------------------------------- #
# Quantum module
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """
    Qiskit wrapper that builds a 1‑qubit variational circuit.
    The circuit consists of an angle‑encoding layer followed by a random
    entangling layer and a final RX/RY rotation.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("aer_simulator")

    def expectation(self, params: np.ndarray) -> np.ndarray:
        """Return the expectation value of Pauli‑Z for each qubit."""
        circ = qiskit.QuantumCircuit(self.n_qubits)
        # Angle encoding
        for i, theta in enumerate(params):
            circ.rx(theta, i)
        # Random entangling layer
        for _ in range(30):
            q1, q2 = np.random.choice(self.n_qubits, 2, replace=False)
            gate = np.random.choice(["cx", "cz", "swap"])
            if gate == "cx":
                circ.cx(q1, q2)
            elif gate == "cz":
                circ.cz(q1, q2)
            else:
                circ.swap(q1, q2)
        # Additional RX/RY rotations
        for i in range(self.n_qubits):
            circ.rx(np.pi / 4, i)
            circ.ry(np.pi / 4, i)
        circ.measure_all()

        compiled = transpile(circ, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        total = sum(counts.values())
        exps = []
        for q in range(self.n_qubits):
            exp = 0.0
            for bitstring, cnt in counts.items():
                bit = bitstring[::-1][q]  # little‑endian
                val = 1.0 if bit == "0" else -1.0
                exp += val * cnt / total
            exps.append(exp)
        return np.array(exps, dtype=np.float32)


class QuantumFunction(torch.autograd.Function):
    """Autograd wrapper that forwards a batch of parameters to Qiskit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit):
        # inputs shape: (batch, n_qubits)
        batch_size, n_qubits = inputs.shape
        out = torch.zeros((batch_size, n_qubits), device=inputs.device, dtype=torch.float32)
        for i in range(batch_size):
            params = inputs[i].cpu().numpy()
            out[i] = torch.tensor(circuit.expectation(params), dtype=torch.float32)
        ctx.save_for_backward(inputs)
        ctx.circuit = circuit
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        circuit: QuantumCircuit = ctx.circuit
        shift = np.pi / 2
        batch_size, n_qubits = inputs.shape
        grad_inputs = torch.zeros_like(inputs)
        for i in range(batch_size):
            params = inputs[i].cpu().numpy()
            grads = np.zeros(n_qubits, dtype=np.float32)
            for j in range(n_qubits):
                params_plus = params.copy()
                params_minus = params.copy()
                params_plus[j] += shift
                params_minus[j] -= shift
                exp_plus = circuit.expectation(params_plus)
                exp_minus = circuit.expectation(params_minus)
                grads[j] = (exp_plus[j] - exp_minus[j]) / 2.0
            grad_inputs[i] = torch.tensor(grads, dtype=torch.float32)
        return grad_inputs * grad_output, None


class QuantumLayer(nn.Module):
    """Wraps the quantum circuit into a differentiable PyTorch module."""

    def __init__(self, num_qubits: int = 1):
        super().__init__()
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(num_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return QuantumFunction.apply(x, self.circuit)


# --------------------------------------------------------------------------- #
# Hybrid regression model
# --------------------------------------------------------------------------- #
class UnifiedQuantumRegression(nn.Module):
    """
    Hybrid regression model that concatenates a residual MLP with a
    1‑qubit variational circuit.  The circuit is differentiable via a
    parameter‑shift implementation and runs on the Aer simulator.
    """

    def __init__(self, num_features: int, num_qubits: int = 1):
        super().__init__()
        self.classical = ClassicalFeatureExtractor(num_features)
        self.quantum = QuantumLayer(num_qubits)
        self.head = nn.Linear(num_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Classical feature extraction
        h = self.classical(x)
        # Pad or truncate to match the quantum circuit input size
        if h.shape[1]!= self.quantum.num_qubits:
            h = nn.functional.pad(
                h,
                (0, self.quantum.num_qubits - h.shape[1]),
                mode="constant",
                value=0.0,
            )
        # Quantum feature extraction
        q = self.quantum(h)
        # Linear head
        out = self.head(q)
        return out.squeeze(-1)


__all__ = [
    "UnifiedQuantumRegression",
    "RegressionDataset",
    "generate_superposition_data",
    "ResidualBlock",
    "ClassicalFeatureExtractor",
    "QuantumCircuit",
    "QuantumFunction",
    "QuantumLayer",
]
