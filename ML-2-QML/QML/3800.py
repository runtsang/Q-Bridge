"""
HybridAutoQCNet – quantum‑enabled implementation.

This module inherits from the classical HybridAutoQCNet and replaces its
classical head with a parameterised quantum expectation head. The
quantum circuit operates on the latent vector produced by the
autoencoder, evaluating the Z‑expectation on the first qubit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli

# Import the classical backbone
from.ml_code import AutoencoderConfig, AutoencoderNet, HybridAutoQCNet as BaseHybridAutoQCNet

# --------------------------------------------------------------------------- #
# Quantum circuit wrapper
# --------------------------------------------------------------------------- #
class QuantumCircuitWrapper:
    """
    A lightweight wrapper around a parameterised Pauli‑evolution circuit
    that measures the expectation value of Z on the first qubit.
    """
    def __init__(self, n_qubits: int, backend=None, shots: int = 200) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots

        # Build a simple variational circuit: H → RY(θ) on each qubit
        self.theta = Parameter("θ")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of parameter sets.
        Parameters are expected to be a 1‑D array of length `n_qubits`.
        Returns the Z‑expectation value of the first qubit.
        """
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: p} for p in params]
        )
        job = self.backend.run(qobj)
        counts = job.result().get_counts()

        zero = sum(v for k, v in counts.items() if k[0] == "0")
        one = sum(v for k, v in counts.items() if k[0] == "1")
        expectation = (zero - one) / self.shots
        return np.array([expectation])

# --------------------------------------------------------------------------- #
# Quantum head
# --------------------------------------------------------------------------- #
class QuantumHybridHead(nn.Module):
    """
    PyTorch wrapper that forwards a latent vector through the quantum circuit
    and returns the expectation value as a probability.
    """
    def __init__(self, in_features: int, backend=None, shots: int = 200) -> None:
        super().__init__()
        self.quantum = QuantumCircuitWrapper(in_features, backend, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x shape: (batch, in_features)
        batch = x.detach().cpu().numpy()
        probs = [self.quantum.run(sample)[0] for sample in batch]
        return torch.tensor(probs, device=x.device).unsqueeze(-1)

# --------------------------------------------------------------------------- #
# HybridAutoQCNet with quantum head
# --------------------------------------------------------------------------- #
class HybridAutoQCNet(BaseHybridAutoQCNet):
    """
    Subclass of the classical HybridAutoQCNet that swaps its head for
    a quantum hybrid head. All other components (convolution, dense,
    autoencoder) remain unchanged, enabling a direct side‑by‑side
    comparison of classical vs. quantum back‑ends.
    """
    def __init__(self) -> None:
        super().__init__()
        # Replace the classical head with a quantum head
        latent_dim = AutoencoderConfig(input_dim=32, latent_dim=8).latent_dim
        self.head = QuantumHybridHead(in_features=latent_dim, shots=400)

__all__ = ["QuantumCircuitWrapper", "QuantumHybridHead", "HybridAutoQCNet"]
