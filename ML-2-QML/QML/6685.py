"""Quantum encoder for the hybrid autoencoder."""
from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator

class QuantumEncoder:
    """A simple variational circuit that encodes a latent vector into a quantum state."""
    def __init__(self, num_qubits: int, shots: int = 1024, backend=None) -> None:
        self.num_qubits = num_qubits
        self.shots = shots
        self.backend = backend or AerSimulator()
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Build a circuit with one Ry gate per qubit, followed by measurement."""
        self.circuit = QuantumCircuit(self.num_qubits)
        # Create a parameter for each qubit
        self.params = [Parameter(f"theta_{i}") for i in range(self.num_qubits)]
        # Apply H to prepare superposition
        self.circuit.h(range(self.num_qubits))
        self.circuit.barrier()
        # Apply Ry with the parameters
        for i, p in enumerate(self.params):
            self.circuit.ry(p, i)
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit for the given parameters and return expectation values of Z."""
        if params.ndim == 1:
            params = params.reshape(1, -1)
        expectations = []
        for p in params:
            # Bind parameters
            bind_dict = {self.params[i]: p[i] for i in range(self.num_qubits)}
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots, parameter_binds=[bind_dict])
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            # Compute expectation value of Z for each qubit
            exp = np.zeros(self.num_qubits)
            for bitstring, count in counts.items():
                for i in range(self.num_qubits):
                    bit = int(bitstring[::-1][i])  # LSB first
                    exp[i] += (1 if bit == 0 else -1) * count
            exp /= self.shots
            expectations.append(exp)
        return np.array(expectations)

def refine_latent(z: torch.Tensor, encoder: QuantumEncoder) -> torch.Tensor:
    """Refine a latent vector using the quantum encoder.
    z: tensor of shape (batch, latent_dim) or (latent_dim,)
    Returns a tensor of the same shape with quantum‑refined values."""
    # Convert to numpy
    z_np = z.detach().cpu().numpy()
    if z_np.ndim == 1:
        z_np = z_np.reshape(1, -1)
    # Run the encoder
    refined_np = encoder.run(z_np)
    # Convert back to torch
    refined = torch.tensor(refined_np, dtype=z.dtype, device=z.device)
    return refined

__all__ = ["QuantumEncoder", "refine_latent"]
