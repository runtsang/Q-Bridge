"""
Hybrid Quantum Binary Classifier – Quantum (Qiskit) counterpart.

Implements a CNN backbone followed by a parametrised two‑qubit circuit
whose expectation value serves as the classification head.
The circuit is built from a random layer and a Ry‑encoding that
accepts the classical features as angles.  Gradient estimation
uses the parameter‑shift rule for efficient back‑propagation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit as QC, QuantumRegister, ClassicalRegister, transpile, assemble
from qiskit.providers.aer import AerSimulator

class RandomParameterizedCircuit:
    """Two‑qubit circuit with a random layer followed by Ry encoding."""
    def __init__(self, n_qubits: int = 2, seed: int | None = None):
        self.n_qubits = n_qubits
        self.qreg = QuantumRegister(n_qubits)
        self.circ = QC(self.qreg)
        rng = np.random.default_rng(seed)
        # Random layer: random single‑qubit rotations and CNOTs
        for q in range(n_qubits):
            theta = rng.uniform(0, 2*np.pi)
            phi   = rng.uniform(0, 2*np.pi)
            lam   = rng.uniform(0, 2*np.pi)
            self.circ.u(theta, phi, lam, q)
        for i in range(n_qubits-1):
            self.circ.cx(i, i+1)
        for i in reversed(range(n_qubits-1)):
            self.circ.cx(i, i+1)

    def encode(self, angles: np.ndarray) -> QC:
        """Attach Ry gates encoding the input angles."""
        circ = QC(self.qreg)
        for q, ang in enumerate(angles):
            circ.ry(ang, q)
        return circ

    def circuit(self, angles: np.ndarray) -> QC:
        """Full circuit: random layer + encoding."""
        circ = self.circ.copy()
        circ.compose(self.encode(angles), inplace=True)
        circ.measure_all()
        return circ

class QuantumHybridFunction(torch.autograd.Function):
    """Bridges PyTorch and the Qiskit circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit_builder: RandomParameterizedCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit_builder = circuit_builder
        angles = inputs.cpu().numpy()
        exp_vals = []
        backend = AerSimulator()
        for ang in angles:
            qc = circuit_builder.circuit(ang)
            qobj = assemble(transpile(qc, backend))
            result = backend.run(qobj).result()
            counts = result.get_counts()
            exp = sum((int(k[0]) * -1 for k in counts)) / sum(counts.values())
            exp_vals.append(exp)
        exp_tensor = torch.tensor(exp_vals, device=inputs.device, dtype=torch.float32)
        ctx.save_for_backward(inputs, exp_tensor)
        return exp_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, exp = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for angle in inputs.cpu().numpy():
            ang_plus  = angle + shift
            ang_minus = angle - shift
            plus_qc = ctx.circuit_builder.circuit(ang_plus)
            minus_qc = ctx.circuit_builder.circuit(ang_minus)
            backend = AerSimulator()
            plus_exp = QuantumHybridFunction._expectation(plus_qc, backend)
            minus_exp = QuantumHybridFunction._expectation(minus_qc, backend)
            grads.append(plus_exp - minus_exp)
        grad_tensor = torch.tensor(grads, device=inputs.device, dtype=torch.float32)
        return grad_tensor * grad_output, None, None

    @staticmethod
    def _expectation(qc: QC, backend: AerSimulator) -> float:
        qobj = assemble(transpile(qc, backend))
        result = backend.run(qobj).result()
        counts = result.get_counts()
        exp = sum((int(k[0]) * -1 for k in counts)) / sum(counts.values())
        return exp

class HybridQuantumHead(nn.Module):
    """Quantum head that outputs a single value via expectation."""
    def __init__(self, shift: float = np.pi/2):
        super().__init__()
        self.shift = shift
        self.circuit_builder = RandomParameterizedCircuit(n_qubits=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(-1)
        return QuantumHybridFunction.apply(x, self.circuit_builder, self.shift)

class HybridQuantumBinaryClassifier(nn.Module):
    """CNN backbone + quantum expectation head."""
    def __init__(
        self,
        in_channels: int = 3,
        conv_features: list[int] | None = None,
        fc_features: list[int] | None = None,
        shift: float = np.pi/2,
        dropout: float = 0.5,
    ):
        super().__init__()
        if conv_features is None:
            conv_features = [6, 15]
        if fc_features is None:
            fc_features = [120, 84]

        # Backbone
        layers = []
        in_ch = in_channels
        for out_ch in conv_features:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=5 if out_ch==6 else 3,
                                    stride=2, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=1))
            layers.append(nn.Dropout2d(p=dropout))
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)

        # Fully connected
        flat_dim = 55815  # for 32x32 RGB
        fc_layers = []
        in_fc = flat_dim
        for out_fc in fc_features:
            fc_layers.append(nn.Linear(in_fc, out_fc))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout))
            in_fc = out_fc
        fc_layers.append(nn.Linear(in_fc, 1))
        self.fcs = nn.Sequential(*fc_layers)

        # Quantum head
        self.quantum_head = HybridQuantumHead(shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fcs(x)
        probs = self.quantum_head(x)
        return torch.stack((probs, 1 - probs), dim=-1)

__all__ = ["RandomParameterizedCircuit", "QuantumHybridFunction",
           "HybridQuantumHead", "HybridQuantumBinaryClassifier"]
