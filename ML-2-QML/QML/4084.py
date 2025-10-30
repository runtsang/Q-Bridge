"""Quantum hybrid model that replaces the classical parts with their quantum counterparts.

This module fuses:
- a quantum quanvolution filter (Pair 1),
- a true quantum expectation head using Qiskit (Pair 2),
- and a quantum regression circuit inspired by EstimatorQNN (Pair 3).

The architecture is identical to the classical version but all quantum operations run on a simulator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import qiskit
from qiskit import assemble, transpile
import numpy as np

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """
    2×2 patch quantum kernel implemented with torchquantum.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
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

class QuantumCircuit:
    """
    Parametrised two‑qubit circuit executed on Qiskit Aer.
    """
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
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
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge between PyTorch and a Qiskit quantum circuit.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift
        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)
        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

class Hybrid(nn.Module):
    """
    Hybrid layer that forwards activations through a quantum circuit.
    """
    def __init__(self, in_features: int, backend, shots: int, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(in_features, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)

class EstimatorQNNQuantum(nn.Module):
    """
    Small quantum regression circuit mirroring the EstimatorQNN example.
    """
    def __init__(self) -> None:
        super().__init__()
        self.params1 = [qiskit.circuit.Parameter("input1"), qiskit.circuit.Parameter("weight1")]
        self.qc1 = qiskit.QuantumCircuit(1)
        self.qc1.h(0)
        self.qc1.ry(self.params1[0], 0)
        self.qc1.rx(self.params1[1], 0)
        self.backend = qiskit.Aer.get_backend("aer_simulator_statevector")
        self.estimator = qiskit.primitives.StatevectorEstimator()
        self.observable = qiskit.quantum_info.SparsePauliOp.from_list([("Y", 1)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input1 = inputs[:, 0].detach().cpu().numpy()
        weight1 = inputs[:, 1].detach().cpu().numpy()
        bound_circuits = [self.qc1.bind_parameters({self.params1[0]: a, self.params1[1]: w}) for a, w in zip(input1, weight1)]
        results = self.estimator.run(bound_circuits, backend=self.backend)
        expectations = []
        for sv in results:
            expectations.append(sv.expectation(self.observable))
        return torch.tensor(expectations, dtype=torch.float32, device=inputs.device).unsqueeze(-1)

class QuanvolutionHybridNet(nn.Module):
    """
    Quantum‑augmented version of QuanvolutionHybridNet.
    Replaces the classical quanvolution filter with a quantum kernel
    and the hybrid head with a genuine quantum expectation layer.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter(in_channels, 4)
        self.fc = nn.Linear(4 * 14 * 14, 120)
        self.dropout = nn.Dropout(0.5)
        self.regressor = EstimatorQNNQuantum()
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(120, backend, shots=100, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        x = F.relu(self.fc(features))
        x = self.dropout(x)
        logits = self.hybrid(x)
        probs = torch.cat([logits, 1 - logits], dim=-1)
        reg = self.regressor(x[:, :2])
        return probs, reg
