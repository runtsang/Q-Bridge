"""
Quantum implementation of the hybrid quanvolution binary classifier.
It mirrors the classical architecture but replaces the final dense head
with a two‑qubit parameterised circuit whose expectation value serves as
the logit.  The quanvolution branch is implemented with torchquantum
to preserve quantum‑kernel characteristics.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile

import torchquantum as tq


class QuantumCircuit:
    """
    Two‑qubit circuit with a single rotation parameter.
    The expectation of Pauli‑Z on the first qubit is returned.
    """
    def __init__(self, backend, shots: int = 1024) -> None:
        self._circuit = qiskit.QuantumCircuit(2)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h([0, 1])
        self._circuit.ry(self.theta, 0)
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
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """
    Autograd wrapper that forwards inputs to the quantum circuit and
    backpropagates finite‑difference gradients.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.detach().cpu().numpy())
        result = torch.tensor(expectation, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.cpu().numpy()) * ctx.shift
        grad = []
        for val, s in zip(inputs.cpu().numpy(), shift):
            right = ctx.circuit.run([val + s])[0]
            left = ctx.circuit.run([val - s])[0]
            grad.append(right - left)
        grad = torch.tensor(grad, dtype=grad_output.dtype, device=grad_output.device)
        return grad * grad_output, None, None


class Hybrid(nn.Module):
    """
    Quantum hybrid layer that maps a scalar feature to a logit via the
    expectation value of the two‑qubit circuit.
    """
    def __init__(self, backend, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Accepts a batch of scalars
        return HybridFunction.apply(inputs.squeeze(), self.circuit, self.shift)


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """
    Quantum quanvolution filter operating on 2×2 patches of a single‑channel
    image.  Each patch is encoded into a 4‑qubit state, processed by a
    random layer, and measured in the Pauli‑Z basis.
    """
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
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class HybridQuanvolutionBinaryClassifier(nn.Module):
    """
    Quantum hybrid network:
        * Convolutional backbone (identical to the classical version)
        * Quantum quanvolution branch
        * Concatenation of both feature vectors
        * Quantum hybrid head producing a single logit
        * Sigmoid activation to obtain binary probabilities
    """
    def __init__(self) -> None:
        super().__init__()
        # Backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        # Quantum quanvolution
        self.quanvolution = QuantumQuanvolutionFilter()
        # Quantum hybrid head
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(backend, shots=512, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Backbone
        xb = F.relu(self.conv1(x))
        xb = self.pool(xb)
        xb = self.drop1(xb)
        xb = F.relu(self.conv2(xb))
        xb = self.pool(xb)
        xb = self.drop1(xb)
        xb = torch.flatten(xb, 1)
        xb = F.relu(self.fc1(xb))
        xb = self.drop2(xb)
        xb = F.relu(self.fc2(xb))
        # Quantum quanvolution branch
        # Convert RGB to grayscale for the quantum filter
        qx = x.mean(dim=1, keepdim=True)
        qb = self.quanvolution(qx)
        # Concatenate
        combined = torch.cat((xb, qb), dim=1)
        # Use the mean of the combined vector as the input to the quantum head
        mean_feature = combined.mean(dim=1, keepdim=True)
        logits = self.hybrid(mean_feature)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridQuanvolutionBinaryClassifier", "QuantumQuanvolutionFilter"]
