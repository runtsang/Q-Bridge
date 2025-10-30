"""
Quantum autoencoder implemented with PennyLane.  The circuit encodes
data into a small latent subspace and reconstructs it via a swap‑test
fidelity loss, which is maximised during training.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer
from typing import Tuple, List


class AutoencoderExtended:
    """
    Quantum autoencoder that maps an input state to a latent register and
    reconstructs it.  The model contains two sub‑circuits:
        * ``encoder`` – maps |ψ⟩ → |0⟩^k ⊗ |z⟩
        * ``decoder`` – maps |0⟩^k ⊗ |z⟩ → |ψ̂⟩

    The fidelity between |ψ⟩ and |ψ̂⟩ is estimated with a swap‑test.
    """

    def __init__(self, num_qubits: int, latent_dim: int, num_layers: int = 3) -> None:
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)

        # Trainable parameters for encoder/decoder ansatzes
        self.encoder_weights = pnp.random.randn(num_layers, num_qubits, 3)
        self.decoder_weights = pnp.random.randn(num_layers, num_qubits, 3)
        self.encoder_bias = pnp.random.randn(num_qubits)
        self.decoder_bias = pnp.random.randn(num_qubits)

    def _ansatz(self, params, wires):
        """Generic rotation‑only ansatz."""
        for i in range(self.num_layers):
            qml.Rot(params[i, wires[0], 0],
                    params[i, wires[0], 1],
                    params[i, wires[0], 2],
                    wires=wires[0])
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])

    @qml.qnode(dev)
    def encode(self, input_state: np.ndarray, encoder_params) -> np.ndarray:
        """Encode an input state to the latent register."""
        qml.StatePrep(input_state, wires=range(self.num_qubits))
        self._ansatz(encoder_params, range(self.num_qubits))
        return qml.state()

    @qml.qnode(dev)
    def decode(self, latent_state: np.ndarray, decoder_params) -> np.ndarray:
        """Decode a latent state back to the full space."""
        # Prepare latent state on the first `latent_dim` qubits
        prep = np.zeros(2 ** self.num_qubits, dtype=complex)
        # Place latent qubits in the subspace |z⟩|0⟩^rest
        index = int("".join(["1" if i < self.latent_dim and bit else "0"
                             for i, bit in enumerate(latent_state)]), 2)
        prep[index] = 1.0
        qml.StatePrep(prep, wires=range(self.num_qubits))
        self._ansatz(decoder_params, range(self.num_qubits))
        return qml.state()

    def fidelity(self, psi: np.ndarray, phi: np.ndarray) -> float:
        """Estimate fidelity via swap test."""
        @qml.qnode(self.dev)
        def swap_test():
            qml.Hadamard(wires=0)
            qml.CSWAP(wires=[0, 1, 2])  # 0: ancilla, 1: psi, 2: phi
            qml.Hadamard(wires=0)
            return qml.measure(wires=0, readout="probabilities")[0]
        prob = swap_test()
        return 1 - 2 * prob  # fidelity = 1 - 2*prob(ancilla=1)

    def loss(self, input_state: np.ndarray) -> float:
        """Negative fidelity (to be minimised)."""
        latent = self.encode(input_state, self.encoder_weights)
        recon = self.decode(latent, self.decoder_weights)
        return -self.fidelity(input_state, recon)

    def train(
        self,
        training_data: List[np.ndarray],
        *,
        epochs: int = 200,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> List[float]:
        """Train the autoencoder using Adam optimisation."""
        opt = AdamOptimizer(lr)
        losses: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for x in training_data:
                # Compute gradients
                grads_enc, grads_dec = opt.gradients(
                    lambda enc, dec: self.loss(x),
                    self.encoder_weights,
                    self.decoder_weights,
                )
                # Update parameters
                self.encoder_weights = opt.apply_gradients(
                    self.encoder_weights, grads_enc
                )
                self.decoder_weights = opt.apply_gradients(
                    self.decoder_weights, grads_dec
                )
                epoch_loss += self.loss(x)
            epoch_loss /= len(training_data)
            losses.append(epoch_loss)
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.4f}")
        return losses


__all__ = ["AutoencoderExtended"]
