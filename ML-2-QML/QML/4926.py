"""Quantum implementation of the hybrid Quanvolution model.

This module builds on the classical filter from the original
Quanvolution and replaces the dense head with a parameterised
quantum circuit implemented with Qiskit.  The quantum circuit
acts as an expectation layer, providing a differentiable
interface to the rest of the network.  The design is fully
compatible with the classical version, enabling side‑by‑side
experiments.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.primitives import Sampler as QiskitSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator


class QuantumCircuitWrapper:
    """Executable parametrised two‑qubit circuit used as the hybrid head."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = QuantumCircuit(n_qubits)
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self.theta = self._circuit.decl_parameter("theta")
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict: dict) -> float:
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.squeeze().tolist())
        out = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.squeeze().tolist()) * ctx.shift
        grads = []
        for val, s in zip(inputs.squeeze().tolist(), shift):
            grads.append(
                ctx.circuit.run([val + s])[0] - ctx.circuit.run([val - s])[0]
            )
        grads = torch.tensor(grads, device=inputs.device, dtype=torch.float32)
        return grads * grad_output, None, None, None

class Hybrid(nn.Module):
    """Quantum hybrid head that forwards activations through a variational circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape!= torch.Size([1, 1]):
            inputs = inputs.squeeze()
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuanvolutionHybrid(nn.Module):
    """Hybrid model that fuses the classical filter with a quantum head."""

    def __init__(self, num_classes: int = 10, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.fc1 = nn.Linear(4 * 14 * 14, 1)
        backend = AerSimulator()
        self.hybrid = Hybrid(1, backend, shots=100, shift=shift)
        self.fc2 = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        hidden = self.fc1(features)
        hybrid_out = self.hybrid(hidden)
        logits = self.fc2(hybrid_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuantumCircuitWrapper", "HybridFunction", "Hybrid", "QuanvolutionFilter", "QuanvolutionHybrid"]
