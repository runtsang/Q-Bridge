"""Hybrid EstimatorQNN combining classical feature extraction with a quantum expectation head.

The module defines a single EstimatorQNN class that can be used for both regression and binary classification.
It integrates:
  • A deep feed‑forward network for feature extraction (inspired by the original EstimatorQNN seed).
  • A Pennylane quantum circuit that accepts a single angle (derived from the classical features) and returns the expectation
    of Pauli‑Z. The circuit is differentiable via the parameter‑shift rule.
  • An optional Qiskit implementation for users who prefer the Aer simulator.
  • A HybridFunction autograd node that bridges PyTorch and the quantum backend.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# Optional: import qiskit for fallback
try:
    from qiskit import QuantumCircuit as QC, Aer, transpile, assemble
    from qiskit.providers.aer import AerSimulator
    HAS_QISKIT = True
except Exception:
    HAS_QISKIT = False

# Pennylane imports
try:
    import pennylane as qml
    HAS_PENNYLANE = True
except Exception:
    HAS_PENNYLANE = False


class QuantumNode:
    """Variational circuit executed on a Pennylane device.

    The circuit consists of a single Ry rotation on each qubit followed by a
    CNOT chain and measurement of Pauli‑Z on the first qubit. The circuit
    accepts a single parameter (the sum of classical features) and returns
    the expectation value of Z.
    """
    def __init__(self, n_qubits: int, device_name: str = "default.qubit", shots: int = 1024):
        if not HAS_PENNYLANE:
            raise ImportError("Pennylane is required for QuantumNode.")
        self.n_qubits = n_qubits
        self.device = qml.device(device_name, wires=n_qubits, shots=shots)
        self._node = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self, theta: torch.Tensor):
        for i in range(self.n_qubits):
            qml.RY(theta, wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    def run(self, theta: torch.Tensor) -> torch.Tensor:
        """Execute the circuit for the given angle."""
        return self._node(theta)


class QiskitCircuit:
    """Fallback implementation using Qiskit Aer simulator."""
    def __init__(self, n_qubits: int, shots: int = 1024):
        if not HAS_QISKIT:
            raise ImportError("Qiskit is required for QiskitCircuit.")
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit = QC(n_qubits)
        self.theta = QC.Parameter("theta")

        # Build simple ansatz
        for i in range(n_qubits):
            self.circuit.ry(self.theta, i)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, theta: np.ndarray) -> np.ndarray:
        """Execute the circuit and return expectation of Pauli‑Z."""
        bound_circuit = self.circuit.bind_parameters({self.theta: theta})
        compiled = transpile(bound_circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        # Map |0> -> +1, |1> -> -1
        return np.sum((1 - 2 * states) * probs)


class HybridFunction(Function):
    """Autograd wrapper that forwards a scalar through the quantum circuit
    using the parameter‑shift rule for gradients.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Forward expectation
        angles = inputs.detach().cpu().numpy().flatten()
        exp_vals = np.array([circuit.run(a) for a in angles])
        ctx.save_for_backward(inputs, torch.tensor(exp_vals, device=inputs.device))
        return torch.tensor(exp_vals, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        angles = inputs.detach().cpu().numpy().flatten()
        exp_plus = np.array([ctx.circuit.run(a + shift) for a in angles])
        exp_minus = np.array([ctx.circuit.run(a - shift) for a in angles])
        grad = (exp_plus - exp_minus) / (2 * shift)
        return grad_output * torch.tensor(grad, device=grad_output.device), None, None


class EstimatorQNN(nn.Module):
    """Hybrid estimator that fuses a classical neural network with a quantum expectation head.

    Parameters
    ----------
    in_features : int
        Dimensionality of the input data.
    hidden_sizes : list[int], optional
        Sizes of hidden layers in the classical extractor.
    n_qubits : int, default 2
        Number of qubits used in the quantum circuit.
    backend : str, default "pennylane"
        Which backend to use: "pennylane" or "qiskit".
    shots : int, default 1024
        Number of shots for the quantum backend.
    shift : float, default np.pi/2
        Shift used in the parameter‑shift rule.
    task : str, default "regression"
        Either "regression" or "classification".
    """
    def __init__(
        self,
        in_features: int,
        hidden_sizes: list[int] | None = None,
        n_qubits: int = 2,
        backend: str = "pennylane",
        shots: int = 1024,
        shift: float = np.pi / 2,
        task: str = "regression",
    ):
        super().__init__()
        self.task = task
        hidden_sizes = hidden_sizes or [64, 32]
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.extractor = nn.Sequential(*layers)
        self.n_qubits = n_qubits
        self.shift = shift
        self.backend = backend.lower()
        if self.backend == "pennylane":
            if not HAS_PENNYLANE:
                raise RuntimeError("Pennylane backend requested but not available.")
            self.qnode = QuantumNode(n_qubits, shots=shots)
        elif self.backend == "qiskit":
            if not HAS_QISKIT:
                raise RuntimeError("Qiskit backend requested but not available.")
            self.qnode = QiskitCircuit(n_qubits, shots=shots)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        For regression the output is a single scalar expectation value.
        For classification the output is a probability tensor of shape (batch, 2).
        """
        features = self.extractor(x)
        # Reduce to single angle by mean (could be replaced by a learnable linear)
        angle = features.mean(dim=1, keepdim=True)
        # Quantum expectation
        exp_val = HybridFunction.apply(angle, self.qnode, self.shift)
        if self.task == "classification":
            prob = torch.sigmoid(exp_val)
            return torch.cat((prob, 1 - prob), dim=-1)
        return exp_val.squeeze(-1)


__all__ = ["EstimatorQNN", "HybridFunction", "QuantumNode", "QiskitCircuit"]
