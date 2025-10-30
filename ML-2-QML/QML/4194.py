"""Quantum hybrid network that fuses a quantum autoencoder, classical dense layers, and a quantum expectation head.

The architecture first encodes the raw input into a latent vector via a parameterised quantum circuit, then processes the latent representation with classical dense layers, and finally evaluates a parameterised quantum circuit to produce a binary probability. This design allows the quantum component to act as both feature extractor and classifier head, while the dense layers provide a bridge between the quantum and classical worlds.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import assemble, transpile
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumAutoencoder(nn.Module):
    """Simple quantum autoencoder that maps an input vector to a latent representation.

    The circuit applies Ry rotations parameterised by the input angles, entangles the qubits
    with a chain of CX gates, and measures all qubits.  The expectation value of Z for each
    qubit is used as the latent feature.
    """

    def __init__(self, n_latent: int, backend, shots: int = 1024) -> None:
        super().__init__()
        self.n_latent = n_latent
        self.backend = backend
        self.shots = shots

        # Build the circuit template
        self._circuit = qiskit.QuantumCircuit(n_latent)
        self.theta = qiskit.circuit.ParameterVector("theta", n_latent)
        self._circuit.ry(self.theta, range(n_latent))
        for i in range(n_latent - 1):
            self._circuit.cx(i, i + 1)
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a single set of input angles."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta[j]: thetas[j] for j in range(self.n_latent)}],
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        # Compute expectation of Z for each qubit
        expectations = []
        for qubit in range(self.n_latent):
            exp = 0.0
            for state, count in counts.items():
                # state string is reversed relative to qubit order
                if state[self.n_latent - 1 - qubit] == "1":
                    exp -= count
                else:
                    exp += count
            exp /= self.shots
            expectations.append(exp)
        return np.array(expectations, dtype=np.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Batch forward pass."""
        batch_size = inputs.shape[0]
        outputs = []
        for i in range(batch_size):
            theta_vals = inputs[i, :self.n_latent].detach().cpu().numpy()
            outputs.append(self.run(theta_vals))
        return torch.tensor(outputs, device=inputs.device, dtype=torch.float32)


class QuantumHead(nn.Module):
    """Parameterised quantum circuit that outputs a single expectation value used as a probability."""

    def __init__(self, n_qubits: int, backend, shots: int = 1024, shift: float = 0.0) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.shift = shift

        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        # Expectation of Z on first qubit
        exp = 0.0
        for state, count in counts.items():
            if state[0] == "1":
                exp -= count
            else:
                exp += count
        exp /= self.shots
        return exp

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Batch forward pass."""
        batch_size = inputs.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(self.run([inputs[i].item()]))
        return torch.tensor(outputs, device=inputs.device, dtype=torch.float32)


class HybridAutoencoderQCNet(nn.Module):
    """Hybrid network that fuses a quantum autoencoder, classical dense layers, and a quantum head."""

    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()

        backend = qiskit.Aer.get_backend("aer_simulator")
        self.autoencoder = QuantumAutoencoder(latent_dim, backend, shots=1024)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 120),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(84, 1),
        )

        self.head = QuantumHead(1, backend, shots=1024, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid network."""
        # Encode
        latent = self.autoencoder(x)

        # Dense layers
        logits = self.fc(latent).squeeze(-1)

        # Quantum head
        probs = []
        for logit in logits.tolist():
            probs.append(self.head.run([logit]))
        probs = torch.tensor(probs, device=x.device, dtype=torch.float32)

        return torch.cat((probs.unsqueeze(-1), (1 - probs).unsqueeze(-1)), dim=-1)

__all__ = ["HybridAutoencoderQCNet"]
