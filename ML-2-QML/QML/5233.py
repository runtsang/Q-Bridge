"""Hybrid quantum‑classical binary classifier.

This module implements a quantum‑enhanced version of :class:`HybridBinaryClassifier`.  
The classical CNN backbone is identical to the pure‑classical model, but the final
classification head is replaced by a parameterised quantum circuit inspired by the
QCNN ansatz.  The circuit encodes the reduced feature vector into qubit rotations,
applies a trainable entangling layer, and measures the expectation value of Pauli‑Z
on the first qubit.  A differentiable PyTorch autograd wrapper allows end‑to‑end
training with gradient‑based optimisers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, Pauli

# ---------- Quantum utilities ----------
class QuantumExpectation:
    """Wrapper that runs a parameterised circuit and returns the expectation of Z on qubit 0."""
    def __init__(self, n_qubits: int, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Build a QCNN‑style circuit with feature encoding and entangling gates."""
        theta = ParameterVector("θ", length=self.n_qubits)
        qc = QuantumCircuit(self.n_qubits)

        # Feature encoding with Y‑rotations
        for i in range(self.n_qubits):
            qc.ry(theta[i], i)

        # Simple entangling layer: CNOT chain and a fixed RZZ rotation
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rzz(np.pi / 4, i, i + 1)
            qc.cx(i, i + 1)

        return qc

    def run(self, params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of parameter vectors.

        Parameters
        ----------
        params : np.ndarray of shape (batch, n_qubits)
            Parameters to bind to the circuit.

        Returns
        -------
        np.ndarray of shape (batch,)
            Expectation value of Pauli‑Z on qubit 0 for each parameter set.
        """
        expectations = []
        for p in params:
            qc_bound = self.circuit.copy()
            bind_dict = {self.circuit.parameters[i]: p[i] for i in range(self.n_qubits)}
            qc_bound = qc_bound.bind_parameters(bind_dict)
            sv = Statevector.from_instruction(qc_bound)
            expectation = sv.expectation_value(Pauli("Z" + "I" * (self.n_qubits - 1)))
            expectations.append(expectation)
        return np.array(expectations)

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumExpectation, shift: float = 0.0) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.save_for_backward(inputs)
        params = inputs.detach().cpu().numpy()
        exp_vals = ctx.circuit.run(params)
        return torch.tensor(exp_vals, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        eps = 1e-3
        params = inputs.detach().cpu().numpy()
        grad = []
        for i in range(params.shape[1]):
            pert_plus = params.copy()
            pert_minus = params.copy()
            pert_plus[:, i] += eps
            pert_minus[:, i] -= eps
            exp_plus = ctx.circuit.run(pert_plus)
            exp_minus = ctx.circuit.run(pert_minus)
            grad.append((exp_plus - exp_minus) / (2 * eps))
        grad = np.stack(grad, axis=1)
        grad = torch.tensor(grad, dtype=inputs.dtype, device=inputs.device)
        return grad * grad_output.unsqueeze(1), None, None

class HybridLayer(nn.Module):
    """Quantum layer that maps a reduced feature vector to a scalar output."""
    def __init__(self, n_qubits: int = 8, shots: int = 1024, shift: float = 0.0) -> None:
        super().__init__()
        self.circuit = QuantumExpectation(n_qubits, shots=shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expectation value is 1‑dimensional
        return HybridFunction.apply(inputs, self.circuit, self.shift)

# ---------- Hybrid classifier ----------
class HybridBinaryClassifier(nn.Module):
    """
    Hybrid quantum‑classical binary classifier.

    Architecture:
        - Classical CNN backbone identical to the pure‑classical model.
        - Linear reduction layer that maps the flattened feature map to 8 values.
        - Quantum hybrid layer (QCNN‑style) producing a single expectation value.
        - Sigmoid activation to obtain probability for class 1.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2, n_qubits: int = 8) -> None:
        super().__init__()
        # Classical backbone
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Linear reduction to match qubit count
        self.reducer = nn.Linear(32 * 4 * 4, n_qubits)
        # Quantum head
        self.quantum = HybridLayer(n_qubits=n_qubits, shots=1024, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.reducer(x)
        q_out = self.quantum(x)  # scalar expectation
        probs = torch.sigmoid(q_out)  # probability of class 1
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridBinaryClassifier"]
