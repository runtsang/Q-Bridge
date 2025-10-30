import pennylane as qml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

@dataclass
class AutoencoderConfig:
    """Configuration for the quantum autoencoder."""
    input_dim: int
    latent_dim: int = 4
    device: str = "default.qubit"
    shots: int = 1024
    lr: float = 0.01
    epochs: int = 200

class Autoencoder__gen155:
    """A variational quantum autoencoder implemented with Pennylane."""
    def __init__(self, config: AutoencoderConfig) -> None:
        self.config = config
        self.dev = qml.device(config.device, wires=config.input_dim)
        self.params = nn.Parameter(torch.randn(config.input_dim, 3))
        self.input_wires = list(range(config.input_dim))

        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            # Encode classical data as rotation angles
            for i in self.input_wires:
                qml.RX(x[i], wires=i)
                qml.RZ(x[i], wires=i)
            # Variational layers
            for layer in range(self.params.shape[0]):
                for i in self.input_wires:
                    qml.RX(self.params[layer, i], wires=i)
                    qml.RZ(self.params[layer, i], wires=i)
                for i in range(len(self.input_wires)-1):
                    qml.CNOT(self.input_wires[i], self.input_wires[i+1])
            # Return expectation values of PauliZ as the reconstructed data
            return [qml.expval(qml.PauliZ(i)) for i in self.input_wires]
        self.circuit = circuit

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode classical inputs into latent quantum states."""
        # Scale inputs to [-1, 1] for rotation angles
        x = 2 * inputs - 1
        return self.circuit(x)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent states back to classical space."""
        # Map expectation values from [-1, 1] to [0, 1]
        return (latents + 1) / 2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass."""
        latents = self.encode(inputs)
        return self.decode(latents)

    def loss(self, inputs: torch.Tensor) -> torch.Tensor:
        """Mean squared error between inputs and reconstructions."""
        recon = self.forward(inputs)
        return torch.mean((recon - inputs) ** 2)

    def train(self, data: torch.Tensor, *, epochs: int = None, lr: float = None) -> List[float]:
        """Train the quantum autoencoder using Adam."""
        epochs = epochs or self.config.epochs
        lr = lr or self.config.lr
        opt = torch.optim.Adam([self.params], lr=lr)
        history: List[float] = []
        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(torch.float32)
                opt.zero_grad()
                loss = self.loss(batch)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

    def evaluate(self, data: torch.Tensor) -> float:
        """Return mean reconstruction error on the given data."""
        return self.loss(data).item()

__all__ = ["Autoencoder__gen155"]
