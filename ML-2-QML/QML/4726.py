"""QuantumHybridFusion: Quantum‑enabled variant of the hybrid network.

The quantum module exposes the same public class name as the classical
implementation but replaces the final head with a differentiable
parameterised quantum circuit.  The architecture mirrors the
`` QuantumHybridFusion`` class in the classical module, ensuring that
the two can be swapped without changing the API.

The implementation uses Qiskit Aer for simulation.  A custom
``HybridFunction`` implements the parameter‑shift rule so that the
quantum expectation can be differentiated with respect to the
classical feed‑forward weights.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile

__all__ = ["QuantumHybridFusion"]


class QuantumCircuit:
    """Simple parametrised 1‑qubit circuit used as the expectation head.

    The circuit consists of an H gate, a parameterised RY rotation and
    measurement in the computational basis.  It is executed on the Qiskit
    Aer simulator.
    """

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, qubits)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of angles.

        Args:
            thetas: 1‑D array of rotation angles.

        Returns:
            1‑D array of expectation values.
        """
        compiled = transpile(self.circuit, self.backend)
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
    """Differentiable interface to the quantum circuit.

    The forward pass evaluates the expectation value for each sample.
    The backward pass implements the parameter‑shift rule, which is
    compatible with the quantum circuit used in the head.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Convert to NumPy for the simulator
        angles = inputs.detach().cpu().numpy().flatten()
        expectation = ctx.circuit.run(angles)
        result = torch.tensor(expectation, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        angles = inputs.detach().cpu().numpy().flatten()
        grads = []
        for angle in angles:
            right = ctx.circuit.run([angle + shift])[0]
            left = ctx.circuit.run([angle - shift])[0]
            grads.append(right - left)
        grads = torch.tensor(grads, device=inputs.device, dtype=inputs.dtype)
        return grads * grad_output, None, None


class HybridLayer(nn.Module):
    """Quantum expectation head that replaces the classical logistic head.

    The layer accepts a batch of feature vectors, treats each element
    as a rotation angle for the quantum circuit, and returns the
    expectation values.  The layer is fully differentiable thanks to
    ``HybridFunction``.
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 1,
        backend=None,
        shots: int = 100,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        if backend is None:
            backend = qiskit.Aer.get_backend("aer_simulator")
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten all but the last dimension to 1‑D batch
        batch = x.view(-1)
        return HybridFunction.apply(batch, self.circuit, self.shift)


class QuantumHybridFusion(nn.Module):
    """Hybrid quantum‑classical network for binary classification.

    The network architecture matches the classical ``QuantumHybridFusion``
    but replaces the final head with a quantum expectation layer.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flatten and linear map to a single “angle”
        self.fc = nn.Linear(16 * 8 * 8, 1)  # assumes input 32×32
        # Quantum head
        self.head = HybridLayer(
            in_features=1,
            n_qubits=1,
            backend=qiskit.Aer.get_backend("aer_simulator"),
            shots=100,
            shift=np.pi / 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # Treat each scalar as a rotation angle
        x = self.head(x)
        # Convert expectation value to binary probability
        probs = torch.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)


if __name__ == "__main__":
    # Quick sanity test
    net = QuantumHybridFusion()
    dummy = torch.randn(2, 3, 32, 32)
    out = net(dummy)
    print(out.shape)  # should be (2, 2)
