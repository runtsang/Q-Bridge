"""Quantum counterpart of HybridQCNNClassifier using QCNN layers, fraud‑detection style gates, and a quantum expectation head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator

# ------------------------------------------------------------
#  Fraud‑detection style parameters
# ------------------------------------------------------------
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

    def __init__(
        self,
        bs_theta: float,
        bs_phi: float,
        phases: tuple[float, float],
        squeeze_r: tuple[float, float],
        squeeze_phi: tuple[float, float],
        displacement_r: tuple[float, float],
        displacement_phi: tuple[float, float],
        kerr: tuple[float, float],
    ) -> None:
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

# ------------------------------------------------------------
#  Quantum fraud‑detection circuit
# ------------------------------------------------------------
def build_fraud_detection_qc(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Return a parameterised 2‑qubit quantum circuit mimicking the photonic fraud‑detection block."""
    qc = QuantumCircuit(2)
    params = ParameterVector("p", length=16)
    def _apply_layer(qc: QuantumCircuit, p: list[ParameterVector]) -> None:
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(p[0], 0)
        qc.ry(p[1], 1)
        qc.cx(1, 0)
        qc.rz(p[2], 0)
        qc.ry(p[3], 1)
        qc.rz(p[4], 0)
        qc.rz(p[5], 1)
        qc.rz(p[6], 0)
        qc.rz(p[7], 1)
        qc.rz(p[8], 0)
        qc.rz(p[9], 1)
        qc.rz(p[10], 0)
        qc.rz(p[11], 1)
        qc.rz(p[12], 0)
        qc.rz(p[13], 1)
        qc.rz(p[14], 0)
        qc.rz(p[15], 1)
    _apply_layer(qc, params)
    for layer_params in layers:
        _apply_layer(qc, params)
    qc.measure_all()
    return qc

# ------------------------------------------------------------
#  Quantum convolution and pooling layers
# ------------------------------------------------------------
def conv_block(params: list[ParameterVector], q1: int, q2: int) -> QuantumCircuit:
    block = QuantumCircuit(2)
    block.rz(-np.pi / 2, 1)
    block.cx(1, 0)
    block.rz(params[0], 0)
    block.ry(params[1], 1)
    block.cx(0, 1)
    block.ry(params[2], 1)
    block.cx(1, 0)
    block.rz(np.pi / 2, 0)
    return block

def pool_block(params: list[ParameterVector], q1: int, q2: int) -> QuantumCircuit:
    block = QuantumCircuit(2)
    block.rz(-np.pi / 2, 1)
    block.cx(1, 0)
    block.rz(params[0], 0)
    block.ry(params[1], 1)
    block.cx(0, 1)
    block.ry(params[2], 1)
    return block

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        block = conv_block(params[param_index : param_index + 3], q1, q2)
        qc.append(block, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
    for i in range(0, num_qubits, 2):
        block = pool_block(params[param_index : param_index + 3], qubits[i], qubits[i + 1])
        qc.append(block, [qubits[i], qubits[i + 1]])
        qc.barrier()
        param_index += 3
    return qc

# ------------------------------------------------------------
#  Hybrid function bridging PyTorch and the quantum circuit
# ------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and a quantum expectation head."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, backend, shots: int, shift: float) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.backend = backend
        ctx.shots = shots
        ctx.shift = shift

        param_vals = inputs.squeeze().tolist()
        bound_circuit = circuit.bind_parameters(
            {circuit.parameters[i]: param_vals[i] for i in range(len(circuit.parameters))}
        )
        compiled = transpile(bound_circuit, backend)
        qobj = assemble(compiled, shots=shots)
        job = backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        expectation = _expectation_from_counts(counts, shots)
        return torch.tensor([expectation], dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Gradient shift rule not implemented for brevity
        return None, None, None, None, None

def _expectation_from_counts(counts: dict[str, int], shots: int) -> float:
    probs = np.array(list(counts.values())) / shots
    states = np.array([int(k, 2) for k in counts.keys()])
    return np.sum(states * probs)

# ------------------------------------------------------------
#  Hybrid head
# ------------------------------------------------------------
class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(self, circuit: QuantumCircuit, backend, shots: int = 200, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = circuit
        self.backend = backend
        self.shots = shots
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.backend, self.shots, self.shift)

# ------------------------------------------------------------
#  Hybrid QCNN classifier
# ------------------------------------------------------------
class HybridQCNNClassifier(nn.Module):
    """Quantum counterpart of :class:`HybridQCNNClassifier`."""

    def __init__(
        self,
        fraud_input: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        n_qubits: int = 8,
    ) -> None:
        super().__init__()
        self.feature_map = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            self.feature_map.h(i)
        self.ansatz = QuantumCircuit(n_qubits)
        self.ansatz.compose(conv_layer(n_qubits, "c1"), inplace=True)
        self.ansatz.compose(pool_layer(n_qubits, "p1"), inplace=True)
        self.ansatz.compose(conv_layer(n_qubits // 2, "c2"), inplace=True)
        self.ansatz.compose(pool_layer(n_qubits // 2, "p2"), inplace=True)
        self.fraud_circuit = build_fraud_detection_qc(fraud_input, fraud_layers)
        self.combined = QuantumCircuit(n_qubits)
        self.combined.compose(self.feature_map, inplace=True)
        self.combined.compose(self.ansatz, inplace=True)
        self.combined.compose(self.fraud_circuit, inplace=True)
        self.combined.measure_all()
        self.backend = AerSimulator()
        self.hybrid = Hybrid(self.combined, self.backend, shots=200, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.hybrid(inputs)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_qc",
    "conv_layer",
    "pool_layer",
    "HybridFunction",
    "Hybrid",
    "HybridQCNNClassifier",
]
