import pennylane as qml
import numpy as np
from typing import Sequence, Tuple

class QuantumAutoEncoder:
    """Variational quantum autoencoder implemented with Pennylane."""
    def __init__(
        self,
        num_qubits: int,
        latent_dim: int,
        device_name: str = "default.qubit",
        shots: int = 8192,
    ) -> None:
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.device = qml.device(device_name, wires=num_qubits, shots=shots)

        # Default feature map: RealAmplitudes
        self.feature_map = lambda x: qml.templates.RealAmplitudes(
            x, wires=range(num_qubits)
        )

        # Encoder and decoder ansatzes
        self.encoder_ansatz = lambda p, w: qml.templates.RealAmplitudes(p, wires=w)
        self.decoder_ansatz = lambda p, w: qml.templates.RealAmplitudes(p, wires=w)

        # Trainable parameters
        self.encoder_params = np.random.randn(latent_dim * 2).astype(np.float64)
        self.decoder_params = np.random.randn(latent_dim * 2).astype(np.float64)

        @qml.qnode(self.device, interface="autograd")
        def _qnode(x: Sequence[float], enc_p: np.ndarray, dec_p: np.ndarray) -> float:
            # Encode classical data
            self.feature_map(x)
            # Encode latent representation
            self.encoder_ansatz(enc_p, wires=range(latent_dim))
            # Swap test for fidelity estimation
            ancilla = latent_dim
            qml.Hadamard(wires=ancilla)
            for i in range(latent_dim):
                qml.CSWAP(wires=[ancilla, i, i + latent_dim])
            qml.Hadamard(wires=ancilla)
            # Decode latent representation
            self.decoder_ansatz(dec_p, wires=range(latent_dim))
            # Return expectation value of Z on ancilla (related to fidelity)
            return qml.expval(qml.PauliZ(ancilla))

        self._qnode = _qnode

    def fidelity(self, x: Sequence[float]) -> float:
        """Return fidelity (0–1) between input state and reconstructed state."""
        val = self._qnode(x, self.encoder_params, self.decoder_params)
        return (val + 1.0) / 2.0

    def loss(self, x: Sequence[float]) -> float:
        """Loss is 1 – fidelity."""
        return 1.0 - self.fidelity(x)

    def train(
        self,
        data: Sequence[Sequence[float]],
        *,
        epochs: int = 100,
        lr: float = 0.01,
    ) -> list[float]:
        """Gradient‑based training of encoder and decoder parameters."""
        history: list[float] = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x in data:
                # Compute gradients w.r.t encoder and decoder params
                grad_enc = qml.grad(
                    lambda p: self.loss(x)
                )(self.encoder_params)
                grad_dec = qml.grad(
                    lambda p: self.loss(x)
                )(self.decoder_params)
                # Simple SGD update
                self.encoder_params -= lr * grad_enc
                self.decoder_params -= lr * grad_dec
                epoch_loss += self.loss(x)
            epoch_loss /= len(data)
            history.append(epoch_loss)
        return history

def create_quantum_autoencoder(
    num_qubits: int,
    latent_dim: int,
    device_name: str = "default.qubit",
    shots: int = 8192,
) -> QuantumAutoEncoder:
    return QuantumAutoEncoder(num_qubits, latent_dim, device_name=device_name, shots=shots)

__all__ = [
    "QuantumAutoEncoder",
    "create_quantum_autoencoder",
]
