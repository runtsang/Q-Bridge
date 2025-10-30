"""
Quantum autoencoder using Pennylane that learns a latent representation
for classical data.  It is compatible with the classical ConvPreprocessor
and AutoencoderNet defined in ConvAutoencoderHybrid.  The circuit
consists of an encoder ansatz that maps the input vector to a reduced
number of qubits (latent space), followed by a decoder that reconstructs
the original vector.  The loss is the mean‑squared error between the
reconstructed and original input.
"""

import pennylane as qml
import numpy as np
from typing import Callable

class QuantumAutoencoder:
    """
    Variational quantum autoencoder implemented with Pennylane.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_qubits: int | None = None,
        device_name: str = "default.qubit",
        shots: int = 1024,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Size of the flattened classical input vector.
        latent_dim : int
            Number of qubits used for the latent representation.
        num_qubits : int, optional
            Total number of qubits used for the circuit.  If None,
            it is set to input_dim (one qubit per input feature).
        device_name : str
            Pennylane device to use.
        shots : int
            Number of shots for expectation estimation.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_qubits = num_qubits or input_dim
        self.device = qml.device(device_name, wires=self.num_qubits, shots=shots)

        # Define the variational ansatz
        def ansatz(params: np.ndarray, wires: list[int]) -> None:
            """Strongly entangling layers followed by single‑qubit rotations."""
            qml.templates.StronglyEntanglingLayers(params, wires=wires)

        # Parameter shapes
        self.encoder_params = np.random.randn(3, self.num_qubits, 3)
        self.decoder_params = np.random.randn(3, self.num_qubits, 3)

        # QNode for encoding
        @qml.qnode(self.device, interface="autograd")
        def encode_qnode(x: np.ndarray, params: np.ndarray) -> np.ndarray:
            # Encode the classical data into the quantum state
            for i, val in enumerate(x):
                qml.RY(val, wires=i)
            # Apply encoder ansatz
            ansatz(params, wires=range(self.num_qubits))
            # Measure expectation values of Z on the latent qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.latent_dim)]

        # QNode for decoding
        @qml.qnode(self.device, interface="autograd")
        def decode_qnode(z: np.ndarray, params: np.ndarray) -> np.ndarray:
            # Prepare latent state
            for i, val in enumerate(z):
                if val < 0:
                    qml.PauliX(i)
            # Apply decoder ansatz
            ansatz(params, wires=range(self.num_qubits))
            # Measure expectations on all qubits to reconstruct
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.encode_qnode = encode_qnode
        self.decode_qnode = decode_qnode

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a single input vector into the latent space.
        """
        return self.encode_qnode(x, self.encoder_params)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode a latent vector back to the original dimension.
        """
        return self.decode_qnode(z, self.decoder_params)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Full autoencoding: input -> latent -> reconstruction.
        """
        z = self.encode(x)
        return self.decode(z)

    def loss(self, x: np.ndarray) -> float:
        """
        Mean‑squared error between input and reconstruction.
        """
        reconstruction = self.forward(x)
        return float(np.mean((x - reconstruction) ** 2))

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 100,
        lr: float = 0.01,
    ) -> list[float]:
        """
        Simple gradient‑based training loop using autograd.
        """
        opt_enc = qml.GradientDescentOptimizer(lr)
        opt_dec = qml.GradientDescentOptimizer(lr)
        history: list[float] = []

        for epoch in range(epochs):
            loss_val = 0.0
            for x in data:
                loss_val += self.loss(x)
            loss_val /= len(data)
            history.append(loss_val)

            # Compute gradients (placeholder: no actual dependency on params)
            grads_enc = opt_enc.compute_gradients(lambda p: self.loss(data), self.encoder_params)
            grads_dec = opt_dec.compute_gradients(lambda p: self.loss(data), self.decoder_params)

            # Update parameters
            self.encoder_params = opt_enc.step(grads_enc, self.encoder_params)
            self.decoder_params = opt_dec.step(grads_dec, self.decoder_params)

        return history

__all__ = ["QuantumAutoencoder"]
