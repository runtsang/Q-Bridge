"""Quantum hybrid kernel that embeds data via a variational autoencoder and evaluates a swap‑test kernel."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import qiskit as qk
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


class QuantumAutoencoder:
    """
    Builds a RealAmplitudes circuit that compresses input into a smaller latent subspace.
    Uses a COBYLA optimiser to find parameters that minimise reconstruction error.
    """
    def __init__(self, input_dim: int, latent_dim: int, reps: int = 3):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reps = reps
        self.encoder = RealAmplitudes(input_dim, reps=reps)
        self.decoder = RealAmplitudes(input_dim, reps=reps)
        self.params = None

    def _reconstruct(self, state: Statevector) -> Statevector:
        """Apply decoder to encoded state."""
        return self.decoder.construct_statevector(state.n_qubits, self.params)

    def _objective(self, params: np.ndarray) -> float:
        """Reconstruction loss in the computational basis."""
        self.params = params
        # Encode input (here we use zero‑state for simplicity)
        encoded = self.encoder.construct_statevector(self.input_dim, params)
        decoded = self._reconstruct(encoded)
        # Fidelity with original zero‑state
        fidelity = np.abs(decoded.data[0]) ** 2
        return 1.0 - fidelity  # minimise error

    def train(self, epochs: int = 20, seed: int = 42) -> None:
        algorithm_globals.random_seed = seed
        opt = COBYLA()
        opt.set_options({"maxfev": 500 * epochs})
        opt.minimize(self._objective, np.zeros(self.encoder.num_parameters()))

    def encode(self, x: torch.Tensor) -> Statevector:
        """Encode classical data into a quantum state."""
        if self.params is None:
            raise RuntimeError("Autoencoder not trained. Call ``train`` first.")
        # Map data to angles
        angles = x.numpy()
        return self.encoder.construct_statevector(self.input_dim, self.params, angles)


class QuantumHybridKernelAutoencoder:
    """
    Hybrid quantum kernel that first compresses data via a variational autoencoder,
    then evaluates a swap‑test kernel between the resulting states.
    """
    def __init__(self, input_dim: int, latent_dim: int, reps: int = 3):
        self.autoencoder = QuantumAutoencoder(input_dim, latent_dim, reps=reps)
        self.sampler = StatevectorSampler()
        self.num_qubits = input_dim

    def _swap_test(self, state1: Statevector, state2: Statevector) -> float:
        """Return overlap via swap test circuit."""
        qc = QuantumCircuit(self.num_qubits + 1)
        qc.h(self.num_qubits)
        qc.append(state1.to_instruction(), range(self.num_qubits))
        qc.append(state2.to_instruction(), range(self.num_qubits))
        qc.barrier()
        for i in range(self.num_qubits):
            qc.cswap(self.num_qubits, i, i)
        qc.h(self.num_qubits)
        qc.measure(self.num_qubits, 0)
        result = self.sampler.run(qc, shots=1024).get_counts()
        return 1.0 - 2.0 * result.get("1", 0) / 1024

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute kernel value for two classical samples."""
        st1 = self.autoencoder.encode(x)
        st2 = self.autoencoder.encode(y)
        return self._swap_test(st1, st2)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix for two collections of samples."""
        return np.array([[self.kernel(_as_tensor(x), _as_tensor(y)) for y in b] for x in a])


__all__ = [
    "QuantumAutoencoder",
    "QuantumHybridKernelAutoencoder",
]
