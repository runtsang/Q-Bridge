"""Hybrid quantum‑classical QCNN classifier.

This module implements a QCNN quantum circuit with convolutional and pooling
layers, followed by a single‑qubit expectation head.  The network can be
used as a drop‑in replacement for the classical HybridQCNet, enabling
direct comparison between quantum and classical feature extraction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp


class QuantumCircuitWrapper:
    """Wrapper around a parametrised circuit executed on a simulator."""
    def __init__(self, circuit: QuantumCircuit, backend, shots: int) -> None:
        self.circuit = circuit
        self.backend = backend
        self.shots = shots

    def run(self, parameters: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                         parameter_binds=[{p: val} for p, val in zip(self.circuit.parameters, parameters)])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        if isinstance(result, list):
            return np.array([self._expectation(count) for count in result])
        return np.array([self._expectation(result)])

    @staticmethod
    def _expectation(counts: dict) -> float:
        total = sum(counts.values())
        exp = 0.0
        for state, cnt in counts.items():
            exp += int(state, 2) * cnt
        return exp / total


class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectations = []
        for inp in inputs:
            exp = ctx.circuit.run(inp.tolist())[0]
            expectations.append(exp)
        result = torch.tensor(expectations, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        grads = []
        for x, s in zip(inputs.tolist(), shift):
            right = ctx.circuit.run([x + s])[0]
            left = ctx.circuit.run([x - s])[0]
            grads.append(right - left)
        grad_tensor = torch.tensor(grads, dtype=torch.float32)
        return grad_tensor * grad_output, None, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, circuit: QuantumCircuitWrapper, shift: float = 0.0) -> None:
        super().__init__()
        self.circuit = circuit
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(), self.circuit, self.shift)


def build_qcnn_circuit() -> QuantumCircuit:
    """Constructs the QCNN ansatz described in the reference."""
    # Convolution and pooling primitives
    def conv_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi/2, 0)
        return qc

    def conv_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits//2 * 3)
        for i in range(0, num_qubits, 2):
            sub = conv_circuit(params[i//2*3:(i//2+1)*3])
            qc.append(sub, [i, i+1])
        return qc

    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def pool_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits//2 * 3)
        for i in range(0, num_qubits, 2):
            sub = pool_circuit(params[i//2*3:(i//2+1)*3])
            qc.append(sub, [i, i+1])
        return qc

    # Build ansatz
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8)
    ansatz.compose(feature_map, range(8), inplace=True)
    ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(pool_layer(8, "p1"), range(8), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(pool_layer(4, "p2"), range(4, 8), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(pool_layer(2, "p3"), range(6, 8), inplace=True)
    return ansatz


class HybridQCNet(nn.Module):
    """Quantum QCNN classifier mirroring the classical HybridQCNet."""
    def __init__(self, shift: float = np.pi/2) -> None:
        super().__init__()
        backend = qiskit.Aer.get_backend("aer_simulator")
        circuit = build_qcnn_circuit()
        self.circuit_wrapper = QuantumCircuitWrapper(circuit, backend, shots=100)
        self.hybrid = Hybrid(self.circuit_wrapper, shift=shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # inputs: shape (batch, 8) – feature vector for the ZFeatureMap
        probs = self.hybrid(inputs)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumCircuitWrapper", "HybridFunction", "Hybrid", "HybridQCNet"]
