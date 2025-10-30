"""Hybrid fraud detection model combining QCNN-inspired classical layers and a quantum expectation head.

This module implements a PyTorch model that mirrors the structure of the quantum circuit in
`FraudDetection__gen304.py`'s quantum counterpart.  The model consists of a small
convolutional‑style feed‑forward network followed by a variational quantum head
implemented with Qiskit.  The quantum head is wrapped in a differentiable
`HybridFunction` that allows gradient propagation through the expectation value.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Sequence, Iterable, List

# --------------------------------------------------------------------------- #
#  Differentiable quantum layer
# --------------------------------------------------------------------------- #

class HybridFunction(torch.autograd.Function):
    """Forward‑backward hook that evaluates a parameterised Qiskit circuit
    and returns the expectation of a Pauli‑Z observable.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: "QuantumCircuitWrapper", shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit

        # Evaluate the circuit for each batch element
        expectation = ctx.circuit.run(inputs.tolist())
        result = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        grads = []

        for idx, val in enumerate(inputs.numpy()):
            right = ctx.circuit.run([val + shift[idx]])
            left = ctx.circuit.run([val - shift[idx]])
            grads.append(right - left)

        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Linear layer followed by a differentiable quantum expectation head."""
    def __init__(self, in_features: int, circuit: "QuantumCircuitWrapper", shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.circuit = circuit
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs).squeeze(-1)
        return HybridFunction.apply(logits, self.circuit, self.shift)

# --------------------------------------------------------------------------- #
#  Variational quantum circuit wrapper
# --------------------------------------------------------------------------- #

class QuantumCircuitWrapper:
    """Thin wrapper around a Qiskit variational circuit used by Hybrid."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")

        # Build a simple 2‑qubit ansatz
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of angle schedules."""
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
            states = np.array(list(count_dict.keys()), dtype=float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

# --------------------------------------------------------------------------- #
#  FraudDetectionHybrid model
# --------------------------------------------------------------------------- #

class FraudDetectionHybrid(nn.Module):
    """QCNN‑style feature extractor followed by a quantum expectation head.

    The architecture consists of a sequence of linear layers with Tanh
    activations mimicking the quantum convolutional layers of the
    photonic QCNN.  The final layer feeds into a variational quantum
    circuit implemented with Qiskit.  The model outputs a two‑class
    probability distribution.
    """
    def __init__(
        self,
        n_features: int = 2,
        hidden_dims: Sequence[int] = (64, 32),
        n_qubits: int = 2,
        shots: int = 512,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.feature_extractor = nn.Sequential(*layers)

        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, shots=shots)
        self.quantum_head = Hybrid(in_features=hidden_dims[-1], circuit=self.quantum_circuit, shift=shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(inputs)
        x = self.quantum_head(x)
        # Convert expectation in [-1,1] to probability in [0,1]
        probs = (x + 1) / 2
        return torch.cat((probs, 1 - probs), dim=-1)

# --------------------------------------------------------------------------- #
#  Utility: FastEstimator for batched evaluation
# --------------------------------------------------------------------------- #

class FastEstimator:
    """Wrapper that evaluates a model on a list of parameter sets with optional shot noise."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        inputs: Sequence[Sequence[float]],
        *, shots: int | None = None, seed: int | None = None
    ) -> List[List[float]]:
        model = self.model
        model.eval()
        with torch.no_grad():
            batch = torch.tensor(inputs, dtype=torch.float32)
            outputs = model(batch)
            probs = outputs[:, 0].cpu().numpy().tolist()
            if shots is None:
                return [[p] for p in probs]
            rng = np.random.default_rng(seed)
            noisy = [float(rng.normal(p, max(1e-6, 1 / shots))) for p in probs]
            return [[p] for p in noisy]

__all__ = ["FraudDetectionHybrid", "FastEstimator", "QuantumCircuitWrapper", "Hybrid", "HybridFunction"]
