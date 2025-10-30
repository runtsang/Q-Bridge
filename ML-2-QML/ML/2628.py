"""Hybrid classical‑quantum binary classifier.

The module defines a PyTorch‑compatible neural network that consists of
convolutional feature extraction followed by a quantum head.  The quantum
head can be configured to use either a parameter‑shift expectation value
or a sampling‑based probability distribution.  The design is fully
autograd‑ready and can be trained on a CPU or GPU while the quantum
circuit is executed on a qiskit Aer simulator or a real device.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as QiskitSampler


# --------------------------------------------------------------------------- #
# Quantum primitives
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """A parametrised two‑qubit circuit executed on Aer."""

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
        """Execute the parametrised circuit for the provided angles."""
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


# --------------------------------------------------------------------------- #
# Hybrid layers
# --------------------------------------------------------------------------- #
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
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)


# --------------------------------------------------------------------------- #
# Sampler‑based head (optional)
# --------------------------------------------------------------------------- #
class SamplerHead(nn.Module):
    """Quantum sampler head that returns a probability distribution."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        super().__init__()
        # Build a simple two‑qubit circuit with parameters for inputs and weights
        self.params = ParameterVector("param", 6)
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.ry(self.params[0], 0)
        self.circuit.ry(self.params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.params[2], 0)
        self.circuit.ry(self.params[3], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.params[4], 0)
        self.circuit.ry(self.params[5], 1)
        self.circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.sampler = QiskitSampler(self.backend)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Bind the first two parameters to the input and the rest are trainable
        bind_dict = {self.params[0]: inputs[0].item(),
                     self.params[1]: inputs[1].item()}
        result = self.sampler.run(self.circuit, bind_dict)
        # Convert counts to probability of |1> for the first qubit
        probs = np.zeros(2)
        for bitstring, count in result.items():
            probs[int(bitstring[0])] += count
        probs /= self.shots
        return torch.tensor(probs, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Classical feature extractor
# --------------------------------------------------------------------------- #
class QCNet(nn.Module):
    """Convolutional network followed by a quantum expectation or sampler head."""

    def __init__(self, quantum_mode: str = "expectation") -> None:
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum head
        backend = qiskit.Aer.get_backend("aer_simulator")
        if quantum_mode == "expectation":
            self.quantum_head = Hybrid(
                n_qubits=self.fc3.out_features,
                backend=backend,
                shots=100,
                shift=np.pi / 2,
            )
        elif quantum_mode == "sampler":
            self.quantum_head = SamplerHead(
                n_qubits=2, backend=backend, shots=500
            )
        else:
            raise ValueError("quantum_mode must be 'expectation' or'sampler'")

        self.quantum_mode = quantum_mode

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
        x = self.fc3(x)

        if self.quantum_mode == "expectation":
            x = self.quantum_head(x).T
            return torch.cat((x, 1 - x), dim=-1)
        else:  # sampler
            probs = self.quantum_head(x)
            return probs.unsqueeze(0)  # shape (1,2)


__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "SamplerHead", "QCNet"]
