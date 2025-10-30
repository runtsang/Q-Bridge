"""Hybrid classical‑quantum model for multi‑feature extraction.

The module defines a single class ``HybridQuantumNAT`` that can be used purely
classically (via PyTorch) or with a quantum backend (via Qiskit).  It
inherits from :class:`torch.nn.Module` and contains:

* A CNN backbone identical to the original QCNet but with fewer parameters
  for faster training.
* A fully‑connected head producing 4 logits.
* A differentiable hybrid head that runs a *n*-qubit quantum circuit on
  the logits and returns their expectation values.  The circuit implements
  a parameter‑shift rule for automatic differentiation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#   Quantum circuit wrapper (Qiskit)                                         #
# --------------------------------------------------------------------------- #
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator


class QuantumCircuit:
    """Parametrised 4‑qubit circuit used as the hybrid head.

    The circuit is a simple H‑RY‑measure pattern that can be run on a
    simulator or a real backend.  The function ``run`` accepts a batch
    of angles and returns the expectation value of Pauli‑Z on each qubit.
    """

    def __init__(self, n_qubits: int = 4, backend: str = "aer_simulator", shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = AerSimulator(method="statevector")
        self.shots = shots

        # Build a template circuit with a single Ry gate per qubit
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        all_qubits = list(range(self.n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

    def run(self, angles: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of angles.

        Parameters
        ----------
        angles : np.ndarray
            Shape ``(batch, n_qubits)``.  Each row contains the Ry angle
            for every qubit.

        Returns
        -------
        np.ndarray
            Shape ``(batch, n_qubits)`` – expectation values of Pauli‑Z.
        """
        if angles.ndim == 1:
            angles = angles[None, :]
        expectations = []

        for row in angles:
            bound = self._circuit.assign_parameters({self.theta: row}, inplace=False)
            transpiled = transpile(bound, self.backend)
            qobj = assemble(transpiled, shots=self.shots)
            result = self.backend.run(qobj).result()
            counts = result.get_counts()
            exp = 0.0
            total = 0
            for bitstring, cnt in counts.items():
                val = 1 if bitstring.count("1") % 2 == 0 else -1
                exp += val * cnt
                total += cnt
            expectations.append(exp / total)

        return np.array(expectations)


# --------------------------------------------------------------------------- #
#   Differentiable hybrid layer                                            #
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Autograd wrapper around :class:`QuantumCircuit` using the parameter‑shift rule."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        angles_np = inputs.detach().cpu().numpy()
        expectations = circuit.run(angles_np)
        return torch.from_numpy(expectations).to(inputs.device).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shift = ctx.shift
        circuit = ctx.circuit
        inputs = grad_output.new_zeros_like(grad_output)  # placeholder for shape

        grads = []
        for i in range(inputs.shape[-1]):
            shifted_pos = inputs.clone()
            shifted_neg = inputs.clone()
            shifted_pos[..., i] += shift
            shifted_neg[..., i] -= shift

            pos = circuit.run(shifted_pos.cpu().numpy())
            neg = circuit.run(shifted_neg.cpu().numpy())
            grad = (pos - neg) / 2.0
            grads.append(grad)

        grads = np.stack(grads, axis=-1)
        grad_inputs = torch.from_numpy(grads).to(grad_output.device).float()
        return grad_inputs * grad_output, None, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards classical logits through a quantum circuit."""

    def __init__(self, n_qubits: int = 4, backend: str = "aer_simulator",
                 shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)


# --------------------------------------------------------------------------- #
#   Classical–quantum hybrid model                                         #
# --------------------------------------------------------------------------- #
class HybridQuantumNAT(nn.Module):
    """Classical CNN + quantum hybrid head.

    The architecture mirrors the original QCNet but replaces the final linear
    head with a differentiable quantum expectation head.  The output is a
    4‑dimensional vector that can be interpreted as probabilities or as
    auxiliary features for downstream tasks.
    """

    def __init__(self, in_channels: int = 3, n_qubits: int = 4,
                 backend: str = "aer_simulator", shots: int = 1024):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected head
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_qubits)

        # Hybrid quantum layer
        self.hybrid = Hybrid(n_qubits, backend, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
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

        quantum_out = self.hybrid(x)
        return quantum_out


__all__ = ["HybridQuantumNAT"]
