"""Quantum variational autoencoder using PennyLane.

The circuit embeds classical data via RY rotations, uses a parameterised
RealAmplitudes ansatz for the latent space, and another ansatz to decode
back to an output qubit register.  The loss is a reconstruction MSE plus a
KL‑divergence term that regularises the latent expectation values to a
standard normal distribution.  Training is performed with gradient descent
on a state‑vector simulator.
"""

import numpy as np
import pennylane as qml

class Autoencoder:
    """Variational quantum autoencoder with KL regularisation."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        num_qubits: int | None = None,
        device: str = "default.qubit",
        shots: int = 1024,
        learning_rate: float = 0.01,
    ) -> None:
        """Create a new VQA autoencoder."""
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_qubits = num_qubits or (input_dim + latent_dim)
        self.device = qml.device(device, wires=self.num_qubits, shots=shots)
        self.learning_rate = learning_rate

        # Weight vector: [encoding, decoding] concatenated
        self.weights = np.random.randn(self.latent_dim * 4)  # 4 = 2 reps * 2 ansatzes

        # Pre‑compute wire indices
        self.input_wires = list(range(self.input_dim))
        self.latent_wires = list(range(self.input_dim, self.input_dim + self.latent_dim))
        self.output_wires = self.latent_wires  # reuse latent wires for output

    def circuit(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Quantum circuit returning expectation values for output qubits."""
        @qml.qnode(self.device, interface="autograd")
        def qnode(x_data: np.ndarray, w: np.ndarray) -> np.ndarray:
            # Embed classical data
            for i, wire in enumerate(self.input_wires):
                qml.RY(x_data[i], wires=wire)

            # Encode: RealAmplitudes on latent qubits
            qml.RealAmplitudes(
                w[: self.latent_dim * 2],
                wires=self.latent_wires,
                reps=1,
                skip_last_rot=False,
            )

            # Decode: RealAmplitudes on output qubits (reuse latent wires)
            qml.RealAmplitudes(
                w[self.latent_dim * 2 :],
                wires=self.output_wires,
                reps=1,
                skip_last_rot=False,
            )

            # Measure expectation values of PauliZ on output wires
            return [qml.expval(qml.PauliZ(w)) for w in self.output_wires]

        return qnode(x, weights)

    def loss(
        self,
        x: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
        kl_weight: float = 1.0,
    ) -> float:
        """Compute reconstruction loss plus KL penalty."""
        preds = self.circuit(x, weights)

        # Reconstruction MSE
        recon_loss = np.mean((preds - target) ** 2)

        # Estimate latent distribution from expectation values
        latent_expect = preds  # same as preds
        mu_l = np.mean(latent_expect)
        var_l = np.var(latent_expect) + 1e-6  # avoid log(0)
        kl = 0.5 * (var_l + mu_l ** 2 - 1 - np.log(var_l))

        return recon_loss + kl_weight * kl

    def train(
        self,
        data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        kl_weight: float = 1.0,
    ) -> list[float]:
        """Train the VQA autoencoder with simple gradient descent."""
        history: list[float] = []

        n_samples = data.shape[0]
        indices = np.arange(n_samples)

        for epoch in range(epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_x = data[batch_idx]
                batch_t = data[batch_idx]  # reconstruction target is the input

                # Define loss function for this batch
                def loss_fn(w):
                    return self.loss(batch_x, batch_t, w, kl_weight)

                grad_fn = qml.grad(loss_fn)
                loss_val = loss_fn(self.weights)
                grads = grad_fn(self.weights)

                # Update weights
                self.weights -= self.learning_rate * grads

                epoch_loss += loss_val

            history.append(epoch_loss / (n_samples // batch_size))
        return history

__all__ = ["Autoencoder"]
