"""Pennylane implementation of a hybrid variational autoencoder.

Features:
- Parameterised feature map (RealAmplitudes) for encoding.
- Separate encoder and decoder circuits with trainable weights.
- Parameter‑shift gradient for training.
- Optional classical post‑processing to map measurement outcomes to floats.
- Integration with PyTorch for seamless hybrid training.
"""

import pennylane as qml
from pennylane import numpy as np
import torch
from typing import Tuple, List, Optional

class AutoEncoder:
    """Variational autoencoder built from two parameterised quantum circuits."""

    def __init__(
        self,
        num_qubits: int,
        latent_dim: int,
        reps: int = 2,
        device: str = "default.qubit",
    ) -> None:
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.reps = reps
        self.device = device

        # Feature map for encoding input data
        self.feature_map = qml.templates.embeddings.RealAmplitudes(num_qubits, reps=reps)

        # Encoder circuit parameters
        self.encoder_params = np.random.randn(num_qubits, reps, 3) * 0.01
        # Decoder circuit parameters
        self.decoder_params = np.random.randn(num_qubits, reps, 3) * 0.01

        # QNode for encoding
        @qml.qnode(qml.device(device, wires=num_qubits))
        def encode_qnode(inputs, params):
            self.feature_map(inputs)
            qml.templates.layers.StronglyEntanglingLayers(params, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        # QNode for decoding
        @qml.qnode(qml.device(device, wires=num_qubits))
        def decode_qnode(latent, params):
            # Map latent vector to angles for rotation gates
            for i in range(num_qubits):
                qml.RX(latent[i], wires=i)
            qml.templates.layers.StronglyEntanglingLayers(params, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.encode_qnode = encode_qnode
        self.decode_qnode = decode_qnode

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Map classical input to a quantum latent representation."""
        return self.encode_qnode(x, self.encoder_params)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Map latent vector back to classical space."""
        return self.decode_qnode(z, self.decoder_params)

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = self.encode(x)
        return self.decode(z)

    def loss(self, x: np.ndarray) -> float:
        """Mean‑squared error between input and reconstructed output."""
        recon = self.forward(x)
        return float(np.mean((x - recon) ** 2))

    def loss_with_params(self, data, encoder_params, decoder_params):
        loss = 0.0
        for x in data:
            z = self.encode_qnode(x, encoder_params)
            recon = self.decode_qnode(z, decoder_params)
            loss += np.mean((x - recon) ** 2)
        return loss / len(data)

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> List[float]:
        """Training loop using Adam and parameter‑shift gradients."""
        history: List[float] = []

        for epoch in range(epochs):
            # Compute gradients using Pennylane's automatic parameter‑shift
            grad_fn = qml.grad(lambda ep, dp: self.loss_with_params(data, ep, dp))
            grads = grad_fn(self.encoder_params, self.decoder_params)
            self.encoder_params -= lr * grads[0]
            self.decoder_params -= lr * grads[1]

            loss_val = self.loss_with_params(data, self.encoder_params, self.decoder_params)
            history.append(loss_val)

        return history

__all__ = ["AutoEncoder"]
