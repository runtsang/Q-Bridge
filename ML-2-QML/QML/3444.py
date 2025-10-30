"""Quantum regression dataset and hybrid model using Qiskit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator
from torch.utils.data import Dataset


def generate_quantum_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>."""
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    labels = np.sin(2 * thetas) * np.cos(phis)
    return thetas, labels


class QuantumRegressionDataset(Dataset):
    """Dataset providing angles and regression targets for a quantum circuit."""

    def __init__(self, samples: int, num_wires: int):
        self.thetas, self.labels = generate_quantum_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.thetas)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "thetas": torch.tensor(self.thetas[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumCircuit:
    """Parametrised two‑qubit circuit executed on a Qiskit Aer simulator."""

    def __init__(self, n_qubits: int, backend: qiskit.providers.BaseBackend, shots: int):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for the supplied angles and return the expectation of Pauli‑Z."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()

        def expectation(counts_dict):
            expectations = 0.0
            for bitstring, count in counts_dict.items():
                # Convert bitstring to integer list of bits
                bits = np.array([int(b) for b in bitstring[::-1]])
                # z = +1 for |0>, -1 for |1>
                z_vals = 1 - 2 * bits
                expectations += np.sum(z_vals) * count
            return expectations / (self.shots * len(counts_dict))

        return np.array([expectation(counts)])

    def expectation(self, thetas: np.ndarray) -> np.ndarray:
        """Convenience wrapper returning a 1‑D array of expectations."""
        return self.run(thetas).flatten()


class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper that feeds a parametrised quantum circuit into PyTorch."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.inputs = inputs.detach().clone()
        expectations = circuit.expectation(inputs.detach().cpu().numpy())
        return torch.tensor(expectations, device=inputs.device, dtype=inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs = ctx.inputs
        shift = ctx.shift
        shift_array = shift * torch.ones_like(inputs)
        # Compute forward pass with shifted parameters
        exp_right = ctx.circuit.expectation((inputs + shift_array).cpu().numpy())
        exp_left = ctx.circuit.expectation((inputs - shift_array).cpu().numpy())
        grad_inputs = (exp_right - exp_left) / 2.0
        grad_inputs = torch.tensor(grad_inputs, device=grad_output.device, dtype=grad_output.dtype)
        return grad_inputs * grad_output, None, None


class HybridRegressionModel(nn.Module):
    """Hybrid model that forwards classical angles through a quantum circuit."""

    def __init__(self, n_qubits: int = 2, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        backend = AerSimulator()
        self.quantum_circuit = QuantumCircuit(
            n_qubits=n_qubits, backend=backend, shots=shots
        )
        self.shift = shift
        self.head = nn.Linear(1, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # Ensure thetas is a 1‑D tensor of shape [batch]
        thetas = thetas.view(-1)
        expectations = HybridFunction.apply(thetas, self.quantum_circuit, self.shift)
        return self.head(expectations.unsqueeze(-1)).squeeze(-1)


__all__ = ["HybridRegressionModel", "QuantumRegressionDataset", "generate_quantum_superposition_data"]
