import pennylane as qml
import torch
import numpy as np
from torch.optim import Adam


class Autoencoder:
    """Quantum autoencoder built with PennyLane.

    The encoder maps classical data to a quantum state via an angle‑embedding.
    A RealAmplitudes ansatz produces a latent representation encoded in the
    circuit parameters.  The decoder simply evaluates Pauli‑Z expectation
    values, which are interpreted as the reconstructed data.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        reps: int = 2,
        device: str = "default.qubit",
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reps = reps
        self.dev = qml.device(device, wires=input_dim)

        # Parameters of the RealAmplitudes ansatz
        init_params = np.random.randn(input_dim, reps)
        self.params = torch.nn.Parameter(
            torch.tensor(init_params, dtype=torch.float32)
        )

        # Build the quantum node
        def _circuit(x: torch.Tensor):
            qml.AngleEmbedding(x, wires=range(self.input_dim))
            qml.RealAmplitudes(self.params, wires=range(self.input_dim))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.input_dim)]

        self.circuit = qml.QNode(_circuit, self.dev, interface="torch")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation (circuit parameters)."""
        return self.params.detach()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent parameters back to classical data."""
        # In this simple design we treat the circuit outputs as the decoded data
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the autoencoder on input data."""
        return self.circuit(x)

    def train(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 100,
        lr: float = 0.01,
        weight_decay: float = 0.0,
    ) -> list[float]:
        """End‑to‑end training loop returning loss history."""
        opt = Adam([self.params], lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.MSELoss()
        history: list[float] = []

        for _ in range(epochs):
            opt.zero_grad()
            out = self.forward(data)
            loss = loss_fn(out, data)
            loss.backward()
            opt.step()
            history.append(loss.item())
        return history


__all__ = ["Autoencoder"]
