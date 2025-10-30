import pennylane as qml
import torch
import numpy as np
from dataclasses import dataclass
from typing import Sequence

@dataclass
class QuantumAutoencoderConfig:
    input_dim: int
    latent_dim: int = 2
    hidden_layers: Sequence[int] = (1,)  # number of StronglyEntanglingLayers
    shots: int = 1000
    device: str = "default.qubit"

class Autoencoder:
    """
    Hybrid quantum auto‑encoder using PennyLane.  The circuit encodes a
    classical vector into a quantum state, applies a parameterised
    StronglyEntanglingLayers ansatz on the latent qubits, and then
    decodes back to the original dimensionality.  Training is
    performed with automatic differentiation of the fidelity loss.
    """
    def __init__(self, cfg: QuantumAutoencoderConfig) -> None:
        self.cfg = cfg
        self.num_wires = cfg.input_dim
        self.latent_wires = list(range(cfg.latent_dim))
        self.device = qml.device(cfg.device, wires=self.num_wires, shots=cfg.shots)

        # Build the QNode with Torch interface
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch")

        # Initialise variational parameters
        num_params = self._num_params()
        self.params = torch.randn(num_params, requires_grad=True)

    def _num_params(self) -> int:
        # Each StronglyEntanglingLayers layer uses 3*latent_dim parameters
        num_layers = self.cfg.hidden_layers[0]
        return num_layers * 3 * self.cfg.latent_dim

    def _circuit(self, x: torch.Tensor, params: torch.Tensor):
        # Encode each feature as a rotation on the corresponding qubit
        for i, val in enumerate(x):
            qml.RY(val, wires=i)

        # Apply the variational ansatz on the latent qubits
        num_layers = self.cfg.hidden_layers[0]
        params = params.reshape(num_layers, 3, self.cfg.latent_dim)
        qml.templates.StronglyEntanglingLayers(
            num_layers=num_layers,
            wires=self.latent_wires,
            interface="torch",
        )(params)

        return qml.state()

    def encode(self, x: np.ndarray) -> torch.Tensor:
        """Encode a classical vector into a quantum state vector."""
        tensor = torch.tensor(x, dtype=torch.float32)
        state = self.qnode(tensor, self.params)
        return state

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent state back to the input space (mirror of encode)."""
        reversed_x = z.flip(0)
        return self.encode(reversed_x)

    @staticmethod
    def fidelity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the squared overlap |〈a|b〉|²."""
        return torch.abs(torch.vdot(a, b)) ** 2

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Loss is one minus the fidelity between the input state and the
        reconstructed state after a full encode–decode cycle.
        """
        original = self.encode(x)
        reconstructed = self.decode(original)
        return 1.0 - self.fidelity(original, reconstructed)

    def train(self, data: np.ndarray, *, epochs: int = 200, lr: float = 0.01):
        """
        Stochastic gradient descent on the variational parameters using
        the Adam optimizer provided by PennyLane.
        """
        opt = qml.AdamOptimizer(stepsize=lr)
        X = torch.tensor(data, dtype=torch.float32)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for x in X:
                loss, grads = opt.step_and_cost(self.loss, self.params, x)
                epoch_loss += loss.item()
            epoch_loss /= len(X)
            if epoch % 20 == 0:
                print(f"Epoch {epoch:03d} | loss: {epoch_loss:.6f}")

__all__ = ["Autoencoder", "QuantumAutoencoderConfig"]
