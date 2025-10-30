"""
Quantum implementation of the hybrid quanvolution network.
The network uses a quantum 2×2 filter and a quantum expectation head,
while retaining the same API as the classical counterpart.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
import numpy as np


class QuantumCircuit:
    """Parametrised two‑qubit circuit executed on Aer."""
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


class Hybrid(nn.Module):
    """Quantum head that forwards activations through a parametrised circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)


class QuanvolutionFilter(nn.Module):
    """Quantum 2×2 filter inspired by the original Quanvolution example."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = qiskit.quantum_info.QuantumCircuit(self.n_wires)
        # Simplified encoder: Ry rotations on each qubit
        for wire in range(self.n_wires):
            self.encoder.ry(0.0, wire)  # placeholder; parameters are set during forward

        self.random_layer = qiskit.circuit.random.random_circuit(self.n_wires, 2)
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = 100

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
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
                # Encode data into Ry angles
                circuit = self.encoder.copy()
                for idx, val in enumerate(data[0].tolist()):
                    circuit.ry(val, idx)
                circuit += self.random_layer
                circuit.measure_all()
                qobj = assemble(transpile(circuit, self.backend), shots=self.shots)
                job = self.backend.run(qobj)
                result = job.result().get_counts()
                expectation = sum(int(bit) * count for bit, count in result.items())
                patches.append(torch.tensor(expectation / (self.shots * self.n_wires), device=device))
        return torch.cat(patches, dim=0).unsqueeze(1)


class QuanvolutionHybridClassifier(nn.Module):
    """
    Hybrid network that applies a quantum filter and a quantum head
    to produce binary class probabilities.
    """
    def __init__(
        self,
        use_quantum_filter: bool = True,
        use_quantum_head: bool = True,
        threshold: float = 0.0,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.use_quantum_filter = use_quantum_filter
        self.use_quantum_head = use_quantum_head

        if self.use_quantum_filter:
            self.filter = QuanvolutionFilter()
        else:
            raise NotImplementedError("Classical filter not available in quantum build.")

        if self.use_quantum_head:
            backend = qiskit.Aer.get_backend("aer_simulator")
            self.head = Hybrid(n_qubits=1, backend=backend, shots=100, shift=shift)
        else:
            raise NotImplementedError("Classical head not available in quantum build.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        logits = self.head(features)
        probs = torch.cat([logits, 1 - logits], dim=-1)
        return F.log_softmax(probs, dim=-1)


__all__ = ["QuanvolutionHybridClassifier"]
