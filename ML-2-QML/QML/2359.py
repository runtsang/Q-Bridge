"""Quantum quanvolution‑autoencoder for image classification.

This module implements a hybrid quantum–classical pipeline that
mirrors the classical architecture above.  Each 2×2 patch of a 28×28
image is embedded into a 4‑qubit feature map.  A variational ansatz
(``RealAmplitudes``) is applied on a latent + trash subspace, followed
by a swap‑test that entangles the trash qubits with an auxiliary
qubit.  The measurement of the latent qubits yields a compact
quantum latent vector.  All patch‑level latent vectors are concatenated
and fed into a classical linear classifier.

The implementation relies on Qiskit’s ``SamplerQNN`` to wrap the
parameterised circuit into a differentiable layer that can be
optimised with PyTorch.
"""

import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, RawFeatureVector
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN


class QuanvolutionAutoEncoder(nn.Module):
    """Quantum quanvolution filter + autoencoder + classifier.

    Parameters
    ----------
    patch_size : int
        Size of the image patch to be processed by the quantum circuit.
    latent_dim : int
        Number of latent qubits that form the quantum latent vector.
    trash_dim : int
        Number of trash qubits used in the swap‑test.
    num_classes : int
        Number of target classes for the final classifier.
    """
    def __init__(
        self,
        patch_size: int = 2,
        latent_dim: int = 3,
        trash_dim: int = 2,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.num_classes = num_classes

        # Build a single quantum circuit that will be reused for every patch.
        self.circuit = self._build_circuit()
        # SamplerQNN expects input_params (feature map) and weight_params (ansatz).
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=list(range(4)),          # RawFeatureVector has 4 parameters
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,               # return raw expectation values
            output_shape=latent_dim,
            sampler=Sampler(),
        )
        # Number of patches per image
        self.num_patches = (28 // patch_size) ** 2
        # Classifier on concatenated latent vectors
        self.classifier = nn.Linear(latent_dim * self.num_patches, num_classes)

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the parameterised circuit used by the SamplerQNN."""
        # 4 qubits for the RawFeatureVector (patch values)
        feature = RawFeatureVector(num_qubits=4, params=list(range(4)))
        # Latent + trash qubits
        n_latent = self.latent_dim
        n_trash = self.trash_dim
        n_aux = 1
        n_total = 4 + n_latent + 2 * n_trash + n_aux
        qr = QuantumRegister(n_total)
        cr = ClassicalRegister(1)
        circuit = QuantumCircuit(qr, cr)

        # Embed the patch into the first 4 qubits
        circuit.append(feature, list(range(4)))

        # Variational ansatz on latent + trash qubits
        ansatz = RealAmplitudes(n_latent + n_trash, reps=2)
        circuit.append(ansatz, list(range(4, 4 + n_latent + n_trash)))

        # Swap‑test to entangle trash qubits with an auxiliary qubit
        aux = 4 + n_latent + 2 * n_trash
        circuit.h(aux)
        for i in range(n_trash):
            circuit.cswap(aux, 4 + n_latent + i, 4 + n_latent + n_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])

        # Note: the latent qubits (indices 4.. 4+n_latent-1) are not measured
        # but their expectation values are returned by SamplerQNN.
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class log‑softmax logits."""
        batch_size = x.size(0)
        # Split image into patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches shape: (B, C, H', W', patch_h, patch_w)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size, self.patch_size)
        # Flatten each patch to 4 values
        patches = patches.view(batch_size, -1, 4)

        # Run the quantum circuit for every patch
        latent_vectors = []
        for i in range(patches.size(1)):
            patch = patches[:, i, :]  # shape (B, 4)
            # QNN expects input_params of shape (B, 4)
            q_out = self.qnn(patch)
            latent_vectors.append(q_out)
        # Concatenate all latent vectors
        latent = torch.cat(latent_vectors, dim=1)  # shape (B, num_patches * latent_dim)
        logits = self.classifier(latent)
        return torch.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionAutoEncoder"]
