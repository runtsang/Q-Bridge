"""Hybrid classical‑quantum convolutional network with QCNN quantum head.

The quantum head is a QCNN‑style variational circuit built with Qiskit.
It accepts an 8‑dimensional parameter vector from the fully connected
layer and returns the expectation value of Z on the first qubit,
treated as a probability.  The rest of the network (conv + FC)
mirrors the classical version, enabling direct head‑to‑head
comparisons.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.providers.aer import Aer


# --------------------------------------------------------------------------- #
# QCNN quantum ansatz
# --------------------------------------------------------------------------- #
class QCNNQuantumCircuit:
    """Parametric QCNN circuit used as the quantum head.

    The circuit implements the convolution and pooling layers described
    in the QCNN paper.  It accepts an 8‑dimensional parameter vector
    and returns the expectation value of Z on the first qubit.
    """

    def __init__(self, backend, shots: int = 200) -> None:
        self.backend = backend
        self.shots = shots
        self.circuit = self._build_circuit()

    # ---- Helper circuits ---------------------------------------------------
    def _conv_circuit(self, params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
        qc = QuantumCircuit(len(qubits))
        q1, q2 = qubits
        qc.rz(-np.pi / 2, q2)
        qc.cx(q2, q1)
        qc.rz(params[0], q1)
        qc.ry(params[1], q2)
        qc.cx(q1, q2)
        qc.ry(params[2], q2)
        qc.cx(q2, q1)
        qc.rz(np.pi / 2, q1)
        return qc

    def _pool_circuit(self, params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
        qc = QuantumCircuit(len(qubits))
        q1, q2 = qubits
        qc.rz(-np.pi / 2, q2)
        qc.cx(q2, q1)
        qc.rz(params[0], q1)
        qc.ry(params[1], q2)
        qc.cx(q1, q2)
        qc.ry(params[2], q2)
        return qc

    def _layer(self, num_qubits: int, prefix: str, layer_type: str) -> QuantumCircuit:
        """Build either a convolution or pooling layer."""
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        for a, b in zip(qubits[0::2], qubits[1::2]):
            if layer_type == "conv":
                sub = self._conv_circuit(params[param_index:param_index + 3], [a, b])
            else:
                sub = self._pool_circuit(params[param_index:param_index + 3], [a, b])
            qc.append(sub, [a, b])
            qc.barrier()
            param_index += 3
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the full QCNN ansatz."""
        qc = QuantumCircuit(8)

        # Feature map – simple H gates (could be replaced with ZFeatureMap)
        for q in range(8):
            qc.h(q)

        # First convolution and pooling
        qc.append(self._layer(8, "c1", "conv"), range(8))
        qc.append(self._layer(8, "p1", "pool"), range(8))

        # Second convolution and pooling (on half the qubits)
        qc.append(self._layer(4, "c2", "conv"), range(4))
        qc.append(self._layer(4, "p2", "pool"), range(4))

        # Third convolution and pooling (on quarter the qubits)
        qc.append(self._layer(2, "c3", "conv"), range(2))
        qc.append(self._layer(2, "p3", "pool"), range(2))

        return qc

    # -------------------------------------------------------------------------
    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit with the given parameters."""
        # Bind parameters
        bound = self.circuit.bind_parameters(
            {p: v for p, v in zip(self.circuit.parameters, params)}
        )
        compiled = transpile(bound, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()

        # Expectation of Z on qubit 0
        exp = 0.0
        for bitstring, cnt in counts.items():
            prob = cnt / self.shots
            # bitstring is ordered from most to least significant qubit
            z = 1 if bitstring[-1] == "0" else -1
            exp += z * prob
        return np.array([exp])


# --------------------------------------------------------------------------- #
# Differentiable hybrid layer
# --------------------------------------------------------------------------- #
class QCNNHybridFunction(torch.autograd.Function):
    """Autograd wrapper for the QCNN quantum head."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QCNNQuantumCircuit,
                shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        params = inputs.detach().cpu().numpy()
        exp = ctx.circuit.run(params)
        result = torch.tensor(exp, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.cpu().numpy()) * ctx.shift
        grads = []
        for i, val in enumerate(inputs.cpu().numpy()):
            params_plus = np.array(inputs.cpu().numpy(), copy=True)
            params_minus = np.array(inputs.cpu().numpy(), copy=True)
            params_plus[i] += shift[i]
            params_minus[i] -= shift[i]
            exp_plus = ctx.circuit.run(params_plus)[0]
            exp_minus = ctx.circuit.run(params_minus)[0]
            grads.append((exp_plus - exp_minus) / (2 * shift[i]))
        grad_tensor = torch.tensor(grads, dtype=grad_output.dtype,
                                   device=grad_output.device)
        return grad_tensor * grad_output, None, None


# --------------------------------------------------------------------------- #
# Hybrid model
# --------------------------------------------------------------------------- #
class HybridQCNNNet(nn.Module):
    """Convolutional backbone followed by a QCNN quantum head."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)  # 8‑dimensional feature vector

        # Quantum head
        backend = Aer.get_backend("aer_simulator")
        self.quantum_head = QCNNQuantumCircuit(backend, shots=200)
        self.shift = np.pi / 2

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
        x = self.fc3(x)  # shape (batch, 8)
        probs = QCNNHybridFunction.apply(x, self.quantum_head, self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridQCNNNet"]
