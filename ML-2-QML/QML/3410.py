"""Quantum hybrid autoencoder that merges a quanvolution filter with a variational
latent layer.  The class HybridAutoencoder can be used in a quantum‑classical
training loop.  The quantum part encodes each 2×2 patch of an image into a
set of qubits using RX gates, entangles them with a RealAmplitudes ansatz,
and produces a latent vector via measurement of dedicated latent qubits.
A classical linear decoder reconstructs the image from the latent vector.

This module combines the quantum autoencoder from the first seed with the
quanvolution filter from the second seed, providing a unified hybrid
architecture that can be trained end‑to‑end.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

class HybridAutoencoder:
    """Quantum encoder + classical decoder hybrid autoencoder.

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Shape of the input image (channels, height, width).
    latent_dim : int
        Number of latent qubits used by the variational part.
    patch_size : int
        Size of the square patch in the quanvolution layer.
    shots : int, optional
        Number of shots for the statevector sampler.
    """
    def __init__(self,
                 input_shape: tuple[int, int, int],
                 latent_dim: int = 3,
                 patch_size: int = 2,
                 shots: int = 200):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.shots = shots

        algorithm_globals.random_seed = 42
        self.circuit, self.input_params, self.weight_params = self._build_circuit()
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            interpret=lambda x: x,
            output_shape=latent_dim,
            sampler=self.sampler,
        )
        # Classical decoder: simple linear map from latent to pixel space
        self.decoder_weights = np.random.randn(latent_dim, np.prod(input_shape))

    def _build_circuit(self):
        """Build a circuit that encodes an image via a quanvolution layer and
        returns a latent vector.  The circuit has two sets of parameters:
        * input_params: one per pixel in a patch (used to encode the image)
        * weight_params: the ansatz angles for the variational part.
        """
        ch, h, w = self.input_shape
        ps = self.patch_size
        n_patches = (h // ps) * (w // ps)
        n_input_params = n_patches * (ps ** 2)

        # Parameter vector for pixel encoding
        self.input_params = ParameterVector("x", n_input_params)

        # Quantum registers
        qr = QuantumRegister(n_input_params, name='q')
        latent_qr = QuantumRegister(self.latent_dim, name='latent')
        cr = ClassicalRegister(self.latent_dim, name='c')
        circuit = QuantumCircuit(qr, latent_qr, cr)

        # 1. Encode each pixel as an RX gate on its qubit
        for i in range(n_input_params):
            circuit.rx(self.input_params[i], qr[i])

        # 2. Entangle all pixel qubits with a RealAmplitudes ansatz
        circuit.append(RealAmplitudes(n_input_params, reps=2), qr)

        # 3. Variational latent layer: entangle pixel qubits with latent qubits
        #    via a simple CNOT network
        for i in range(self.latent_dim):
            ctrl = qr[i % n_input_params]
            tgt = latent_qr[i]
            circuit.cx(ctrl, tgt)

        # 4. Measure latent qubits
        for i in range(self.latent_dim):
            circuit.measure(latent_qr[i], cr[i])

        # Weight parameters are the ones from the RealAmplitudes ansatz
        weight_params = [p for p in circuit.parameters if p not in self.input_params]
        return circuit, list(self.input_params), weight_params

    def _bindings_from_data(self, data: np.ndarray):
        """Create a list of parameter bindings for a batch of data."""
        patch_vector = self._extract_patches(data).flatten()
        bind = {p: np.pi * val for p, val in zip(self.input_params, patch_vector)}
        return [bind]

    def _extract_patches(self, data: np.ndarray):
        """Return a flattened array of all patches in the image."""
        ch, h, w = self.input_shape
        ps = self.patch_size
        patches = []
        for i in range(0, h, ps):
            for j in range(0, w, ps):
                patch = data[:, i:i+ps, j:j+ps]
                patches.append(patch.flatten())
        return np.array(patches)

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Run the quantum encoder on a single sample and return the latent vector."""
        binds = self._bindings_from_data(data)
        job = self.sampler.run(self.circuit, param_binds=binds, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        probs = np.zeros(self.latent_dim)
        for key, val in counts.items():
            bits = np.array([int(b) for b in key[::-1]])
            probs += bits * val
        probs /= self.shots
        return probs

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Classical linear decoder mapping latent vector to image."""
        flat = latent @ self.decoder_weights
        return flat.reshape(self.input_shape)

    def forward(self, data: np.ndarray) -> np.ndarray:
        """Full autoencoder: quantum encode + classical decode."""
        latent = self.encode(data)
        return self.decode(latent)

    def loss(self, recon: np.ndarray, target: np.ndarray) -> float:
        """Mean squared error loss."""
        return np.mean((recon - target) ** 2)

__all__ = ["HybridAutoencoder"]
