"""Quantum hybrid binary classifier that fuses a CNN, quantum self‑attention, and a variational head.

The model mirrors the classical architecture but replaces the final linear layer with a
parameterised quantum circuit.  A quantum self‑attention block is applied to a small
subset of features, demonstrating how classical and quantum operations can be interleaved.
The EstimatorQNN primitive is instantiated to show how a ready‑made quantum estimator
can be plugged into the workflow, even though it is not used in the forward pass.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import transpile, assemble, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QuantumSelfAttention:
    """Quantum self‑attention block using a small parameterised circuit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = qiskit.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")

    def _build(self, rot: np.ndarray, ent: np.ndarray) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rot[3 * i], i)
            qc.ry(rot[3 * i + 1], i)
            qc.rz(rot[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(ent[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self, backend, rot: np.ndarray, ent: np.ndarray, shots: int = 512):
        qc = self._build(rot, ent)
        job = execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)

class VariationalCircuit:
    """Two‑qubit variational circuit producing a Y‑expectation value."""
    def __init__(self, backend, shots: int = 200):
        self.circuit = qiskit.QuantumCircuit(2)
        self.theta = Parameter("θ")
        self.circuit.h([0, 1])
        self.circuit.ry(self.theta, [0, 1])
        self.circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: t} for t in thetas])
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        expectation = sum(int(s, 2) * c for s, c in counts.items()) / self.shots
        return np.array([expectation])

class HybridFunction(torch.autograd.Function):
    """Differentiable quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        out = circuit.run(inputs.tolist())
        tensor = torch.tensor(out, dtype=torch.float32)
        ctx.save_for_backward(inputs, tensor)
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs) * ctx.shift
        grads = []
        for val in inputs.tolist():
            right = ctx.circuit.run([val + shift])
            left  = ctx.circuit.run([val - shift])
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum head that replaces the final linear layer."""
    def __init__(self, backend, shots: int = 200, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = VariationalCircuit(backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.shift)

class HybridBinaryClassifier(nn.Module):
    """CNN → quantum self‑attention → fully‑connected → quantum head."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        self.attention = QuantumSelfAttention(n_qubits=4)
        self.fc = nn.Sequential(
            nn.Linear(735 + 1, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(backend, shots=200, shift=np.pi / 2)
        # Instantiate EstimatorQNN for demonstration – not used in forward
        dummy_circuit = qiskit.QuantumCircuit(1)
        self.estimator_qnn = EstimatorQNN(
            circuit=dummy_circuit,
            observables=[SparsePauliOp.from_list([("Y", 1)])],
            input_params=[],
            weight_params=[],
            estimator=StatevectorEstimator()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)  # (batch, 735)
        # quantum self‑attention on a small random slice of the features
        rot = np.random.rand(12)
        ent = np.random.rand(3)
        counts = self.attention.run(qiskit.Aer.get_backend("qasm_simulator"),
                                    rot, ent, shots=512)
        # collapse counts into a single scalar feature
        att_scalar = torch.tensor([sum(int(s, 2) * c for s, c in counts.items()) / 512],
                                  dtype=torch.float32)
        att_scalar = att_scalar.expand(x.shape[0], -1)
        x = torch.cat([x, att_scalar], dim=-1)  # (batch, 736)
        x = self.fc(x)
        q_out = self.hybrid(x[:, :1])  # quantum head on first feature
        probs = torch.sigmoid(q_out)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier"]
