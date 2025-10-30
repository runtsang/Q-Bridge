"""Hybrid classical‑quantum quanvolution network (quantum implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np
import qiskit
from typing import Iterable, Tuple, List

# Quantum quanvolution filter
class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2x2 image patches."""
    def __init__(self) -> None:
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

# Quantum hybrid layer using a variational circuit
class QuantumHybridLayer(nn.Module):
    """Hybrid layer that forwards activations through a parameterised quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, features = inputs.shape
        thetas = inputs.cpu().numpy()
        outputs = []
        for i in range(features):
            theta = thetas[:, i]
            expectation = self.quantum_circuit.run(theta)
            outputs.append(expectation)
        outputs = torch.tensor(outputs, dtype=torch.float32).transpose(0, 1).to(inputs.device)
        return outputs.mean(dim=1, keepdim=True)

# Wrapper for a simple two‑qubit circuit
class QuantumCircuit:
    """Wrapper around a parametrised two‑qubit circuit executed on Aer."""
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
        compiled = qiskit.transpile(self._circuit, self.backend)
        qobj = qiskit.assemble(
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

# Hybrid function bridging PyTorch and quantum circuit
class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
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

# Quantum sampler QNN
class QuantumSamplerQNN(tq.QuantumModule):
    """A simple parameterised quantum circuit for a SamplerQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 2
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=4, wires=[0, 1])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.q_layer(qdev)
        measurement = self.measure(qdev)
        return measurement.view(bsz, 2)

# Quantum build classifier circuit
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[qiskit.QuantumCircuit, Iterable, Iterable, List[qiskit.quantum_info.SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp

    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = qiskit.QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)
    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

# Main hybrid quanvolution network (quantum version)
class HybridQuanvolutionNet(nn.Module):
    """Quantum‑enhanced network: quanvolution filter + quantum hybrid head."""
    def __init__(self, num_classes: int = 2, use_quantum_head: bool = False,
                 shift: float = np.pi / 2, backend=None, shots: int = 100, depth: int = 2) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)
        self.use_quantum_head = use_quantum_head
        if backend is None:
            backend = qiskit.Aer.get_backend("aer_simulator")
        if use_quantum_head:
            self.hybrid = QuantumHybridLayer(4 * 14 * 14, backend, shots, shift=shift)
        else:
            self.hybrid = None
        self.sampler = QuantumSamplerQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        if self.hybrid is not None:
            logits = self.hybrid(features)
        else:
            logits = self.classifier(features)
        sampler_out = self.sampler(logits)
        return F.log_softmax(logits, dim=-1), sampler_out

__all__ = [
    "QuantumQuanvolutionFilter",
    "QuantumHybridLayer",
    "QuantumCircuit",
    "HybridFunction",
    "QuantumSamplerQNN",
    "build_classifier_circuit",
    "HybridQuanvolutionNet",
]
