"""Quantum quanvolution + autoencoder hybrid network.

This module implements a quantum analogue of the classical
`QuanvolutionAutoencoder`.  A RealAmplitudes feature map encodes each
2×2 patch of a 28×28 image into a 4‑qubit register.  The resulting
state is then fed into a variational autoencoder that compresses the
information into a small latent register using a swap‑test with an
auxiliary qubit.  The whole pipeline is wrapped in a
`SamplerQNN`, allowing the model to be trained with standard
gradient‑free optimizers such as COBYLA.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import Sampler as StatevectorSampler

class QuantumQuanvolutionFilter:
    """Encodes 2×2 image patches into a quantum state via a RealAmplitudes
    feature map.  Each patch is represented by 4 qubits."""
    def __init__(self, patch_size: int = 2, qubits_per_patch: int = 4) -> None:
        self.patch_size = patch_size
        self.qubits_per_patch = qubits_per_patch
        self.feature_map = RealAmplitudes(qubits_per_patch, reps=1)

    def _extract_patches(self, image: np.ndarray) -> list[np.ndarray]:
        patches = []
        for r in range(0, 28, self.patch_size):
            for c in range(0, 28, self.patch_size):
                patch = image[r:r+self.patch_size, c:c+self.patch_size].flatten()
                patches.append(patch)
        return patches

    def encode_image(self, image: np.ndarray) -> QuantumCircuit:
        """Return a circuit that encodes the entire image patch‑wise."""
        qc = QuantumCircuit()
        patches = self._extract_patches(image)
        for patch in patches:
            qc.compose(self.encode_patch(patch), qc.qubits, inplace=True)
        return qc

    def encode_patch(self, patch: np.ndarray) -> QuantumCircuit:
        """Return a circuit that encodes a single patch into qubits."""
        qr = QuantumRegister(self.qubits_per_patch)
        qc = QuantumCircuit(qr)
        qc.compose(self.feature_map, qr, inplace=True)
        return qc

class QuantumAutoencoder:
    """Variational autoencoder that compresses a set of qubits into a latent
    register and reconstructs them using a swap‑test."""
    def __init__(self, num_latent: int, num_trash: int) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.total_qubits = num_latent + 2 * num_trash + 1  # +1 auxiliary

    def circuit(self, input_params: list[np.ndarray]) -> QuantumCircuit:
        """Build a circuit that accepts classical parameters for the input
        feature map.  `input_params` should contain a list of arrays,
        each of length `self.total_qubits - 1` (excluding the auxiliary)."""
        qr = QuantumRegister(self.total_qubits)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)

        # Encode the input parameters into the first (num_latent+num_trash) qubits
        for i, param in enumerate(input_params):
            qc.ry(param, qr[i])

        # Variational layer (RealAmplitudes)
        var = RealAmplitudes(self.num_latent + self.num_trash, reps=3)
        qc.compose(var, qr[0:self.num_latent + self.num_trash], inplace=True)

        # Swap‑test with auxiliary qubit
        aux = self.num_latent + 2 * self.num_trash
        qc.h(qr[aux])
        for i in range(self.num_trash):
            qc.cswap(qr[aux], qr[self.num_latent + i],
                     qr[self.num_latent + self.num_trash + i])
        qc.h(qr[aux])
        qc.measure(qr[aux], cr[0])

        return qc

class QuantumQuanvolutionAutoencoder:
    """Full quantum pipeline: quanvolution feature map + variational autoencoder."""
    def __init__(self,
                 patch_size: int = 2,
                 latent_dim: int = 3,
                 trash: int = 2) -> None:
        self.filter = QuantumQuanvolutionFilter(patch_size)
        self.autoencoder = QuantumAutoencoder(latent_dim, trash)
        self.sampler = StatevectorSampler()
        self.qnn = self._build_qnn()

    def _build_qnn(self) -> SamplerQNN:
        """Create a SamplerQNN that runs the full pipeline."""
        # The input parameters are the flattened pixel values of the image,
        # normalised to [0, π] for rotation angles.
        def circuit_builder(params: list[np.ndarray]) -> QuantumCircuit:
            qc = QuantumCircuit()
            # Quanvolution part
            qc.compose(self.filter.encode_image(np.zeros((28, 28))), qc.qubits, inplace=True)
            # Autoencoder part
            qc.compose(self.autoencoder.circuit(params), qc.qubits, inplace=True)
            return qc

        return SamplerQNN(
            circuit=circuit_builder,
            input_params=[],
            weight_params=[],
            interpret=lambda x: x,
            output_shape=(28, 28),
            sampler=self.sampler,
        )

    def forward(self, images: np.ndarray) -> np.ndarray:
        """Return reconstructed images for a batch of 28×28 grayscale images."""
        batch = []
        for img in images:
            # Normalise pixel values to [0, π] for rotation angles
            params = (img.flatten() / 255.0 * np.pi).tolist()
            batch.append(params)
        # Run the sampler QNN
        result = self.qnn.forward(batch)
        # Post‑process: reshape to image shape
        return np.array(result).reshape(-1, 28, 28)

__all__ = [
    "QuantumQuanvolutionFilter",
    "QuantumAutoencoder",
    "QuantumQuanvolutionAutoencoder",
]
