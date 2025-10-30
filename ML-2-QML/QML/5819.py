"""Quantum hybrid autoencoder.

The class mirrors the classical interface but uses a variational
circuit built from the original auto‑encoder design.  It combines a
feature‑encoding layer, a swap‑test based latent extraction, and a
reconstruction sub‑circuit.  The decoder is implemented as a
SamplerQNN that interprets the circuit output as a probability
distribution over the reconstructed states.
"""

from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.quantum_info import Statevector

# ---------------------------------------------------------------------------

def _feature_map(data: np.ndarray, num_qubits: int) -> QuantumCircuit:
    """Encode a classical vector into a quantum state using RY gates."""
    qc = QuantumCircuit(num_qubits)
    for i, val in enumerate(data[:num_qubits]):
        qc.ry(val, i)
    return qc

# ---------------------------------------------------------------------------

def _auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Build the swap‑test based auto‑encoder circuit from the reference."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Variational ansatz on the latent + trash qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    # Swap test
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc

# ---------------------------------------------------------------------------

def _domain_wall(qc: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
    """Apply a domain wall (X gates) between qubits a and b (exclusive)."""
    for i in range(int(b / 2), int(b)):
        qc.x(i)
    return qc

# ---------------------------------------------------------------------------

class HybridAutoencoder:
    """Quantum auto‑encoder that supports encoding, decoding and training."""

    def __init__(
        self,
        input_dim: int = 5,
        latent_dim: int = 3,
        trash_dim: int = 2,
        sampler: Sampler | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.sampler = sampler or Sampler()

        # Build the core circuit
        self.core_circuit = _auto_encoder_circuit(latent_dim, trash_dim)
        # Wrap with SamplerQNN
        self.qnn = SamplerQNN(
            circuit=self.core_circuit,
            input_params=[],
            weight_params=self.core_circuit.parameters,
            sampler=self.sampler,
            interpret=lambda x: x,
            output_shape=latent_dim,
        )

    # -----------------------------------------------------------------------

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode a batch of classical data into latent vectors."""
        latents = []
        for sample in data:
            # Build a circuit that incorporates the feature map
            fm = _feature_map(sample, self.input_dim)
            qc = fm.compose(self.core_circuit)
            # Sample the latent output using the QNN
            result = self.sampler.run(qc).result()
            counts = result.get_counts()
            # Convert measurement counts to expectation value
            exp = sum(int(k, 2) for k in counts) / (len(counts) * 2 ** self.latent_dim)
            latents.append(exp)
        return np.array(latents)

    # -----------------------------------------------------------------------

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Decode latent vectors back to the input space."""
        reconstructions = []
        for latent in latents:
            # Bind latent as parameters to the variational circuit
            bound_qc = self.core_circuit.bind_parameters(
                {p: v for p, v in zip(self.core_circuit.parameters, latent)}
            )
            result = self.sampler.run(bound_qc).result()
            counts = result.get_counts()
            recon = np.array([int(k, 2) for k in list(counts.keys())[:1]])
            reconstructions.append(recon)
        return np.array(reconstructions)

    # -----------------------------------------------------------------------

    def forward(self, data: np.ndarray) -> np.ndarray:
        """Full auto‑encoder: encode then decode."""
        latents = self.encode(data)
        return self.decode(latents)

    # -----------------------------------------------------------------------

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 50,
        lr: float = 1e-3,
        optimizer_cls=COBYLA,
    ) -> list[float]:
        """Train the variational parameters to minimize reconstruction loss."""
        algorithm_globals.random_seed = 42
        opt = optimizer_cls(maxiter=epochs, tol=1e-6)
        history = []

        def loss_fn(params):
            bound_qc = self.core_circuit.bind_parameters(
                {p: v for p, v in zip(self.core_circuit.parameters, params)}
            )
            reconstructions = []
            for sample in data:
                fm = _feature_map(sample, self.input_dim)
                qc = fm.compose(bound_qc)
                result = self.sampler.run(qc).result()
                counts = result.get_counts()
                recon = np.array([int(k, 2) for k in list(counts.keys())[:1]])
                reconstructions.append(recon)
            reconstructions = np.array(reconstructions)

            loss = np.mean((reconstructions - data) ** 2)
            return loss

        init_params = np.random.rand(len(self.core_circuit.parameters))
        opt_result = opt.minimize(loss_fn, init_params)
        history.append(opt_result.fun)
        return history

__all__ = ["HybridAutoencoder"]
