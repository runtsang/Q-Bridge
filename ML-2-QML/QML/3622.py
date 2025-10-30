"""Quantum hybrid auto‑encoder with swap‑test based similarity.

The class `HybridAutoencoder` mirrors the public API of the classical
implementation but internally builds a variational circuit using
Qiskit.  It supports an optional kernel matrix calculation that
emulates the quantum kernel method seed.

The implementation is intentionally lightweight; it uses the
`Sampler` primitive for simulation and can be swapped for any Qiskit
backend.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler

__all__ = ["HybridAutoencoder", "train_q_autoencoder"]

# --------------------------------------------------------------------------- #
# Helper: build the auto‑encoder circuit
# --------------------------------------------------------------------------- #

def _auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Return a circuit that encodes data into a latent sub‑space."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Variational ansatz on the latent + trash qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=3)
    qc.append(ansatz, list(range(num_latent + num_trash)))

    # Swap‑test with an auxiliary qubit
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc

# --------------------------------------------------------------------------- #
# Quantum auto‑encoder class
# --------------------------------------------------------------------------- #

class HybridAutoencoder:
    """Quantum auto‑encoder that mirrors the classical API."""
    def __init__(self, num_latent: int = 3, num_trash: int = 2) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.circuit = _auto_encoder_circuit(num_latent, num_trash)
        self.sampler = Sampler()

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode classical data into the latent sub‑space and return the
        probability of measuring |0> on the auxiliary qubit.
        """
        # Data must match the number of variational parameters
        param_len = self.num_latent + self.num_trash
        if data.shape[-1]!= param_len:
            raise ValueError(f"Expected data of length {param_len}, got {data.shape[-1]}")
        resolver = {f"ry_{i}": float(data[i]) for i in range(param_len)}
        bound_circuit = self.circuit.bind_parameters(resolver)
        result = self.sampler.run(bound_circuit, shots=1024)
        counts = result.get_counts()
        p0 = counts.get("0", 0) / 1024
        return np.array([p0])

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Decode is a placeholder – classical post‑processing of the latent."""
        # In a full VAE setting the decoder would be another variational circuit.
        # Here we simply return the latent vector as a reconstruction.
        return latents

    def forward(self, data: np.ndarray) -> np.ndarray:
        """Run the full auto‑encoder pipeline."""
        lat = self.encode(data)
        return self.decode(lat)

    # ----------------------------------------------------------------------- #
    # Quantum kernel construction
    # ----------------------------------------------------------------------- #

    def kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute a quantum kernel Gram matrix between two data sets."""
        n = a.shape[0]
        m = b.shape[0]
        K = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                # Create a circuit that compares a[i] and b[j] via a swap‑test
                c = _auto_encoder_circuit(self.num_latent, self.num_trash)
                resolver = {
                    **{f"ry_{k}": float(a[i, k]) for k in range(a.shape[1])},
                    **{f"ry_{k}": -float(b[j, k]) for k in range(b.shape[1])},
                }
                bound = c.bind_parameters(resolver)
                result = self.sampler.run(bound, shots=1024)
                counts = result.get_counts()
                p0 = counts.get("0", 0) / 1024
                K[i, j] = p0
        return K

# --------------------------------------------------------------------------- #
# Training helper (placeholder)
# --------------------------------------------------------------------------- #

def train_q_autoencoder(
    model: HybridAutoencoder,
    data: np.ndarray,
    *,
    epochs: int = 10,
    lr: float = 0.01,
) -> list[float]:
    """Placeholder training routine for the quantum auto‑encoder."""
    # In a real setting one would define a cost function on the sampler
    # and use a classical optimiser to update the ansatz parameters.
    # Here we simply return a dummy history.
    return [0.0] * epochs
