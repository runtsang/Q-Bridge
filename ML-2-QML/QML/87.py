"""Pennylane implementation of a variational quantum autoencoder."""
import pennylane as qml
import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class AutoencoderConfig:
    num_qubits: int
    reps: int = 3
    device: str = "default.qubit"

class AutoencoderModel(nn.Module):
    """Variational quantum autoencoder with a classical readout."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.num_qubits = config.num_qubits
        self.dev = qml.device(config.device, wires=self.num_qubits)
        # Trainable rotation parameters for the variational ansatz
        self.params = nn.Parameter(
            torch.randn(config.reps, self.num_qubits, 3) * 0.1
        )

    def _variational_layer(self, params: torch.Tensor) -> None:
        """Apply a single variational layer."""
        for w in range(self.num_qubits):
            qml.Rot(params[0, w, 0], params[0, w, 1], params[0, w, 2], wires=w)
        for w in range(self.num_qubits - 1):
            qml.CNOT(wires=[w, w + 1])

    def _full_ansatz(self, x: torch.Tensor, params: torch.Tensor) -> None:
        """Encode classical data and apply the variational circuit."""
        # Feature map: RY encoding of each input value
        for i, xi in enumerate(x):
            qml.RY(xi, wires=i)
        # Repeated variational layers
        for r in range(params.shape[0]):
            for w in range(self.num_qubits):
                qml.Rot(params[r, w, 0], params[r, w, 1], params[r, w, 2], wires=w)
            for w in range(self.num_qubits - 1):
                qml.CNOT(wires=[w, w + 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return reconstructed data as expectation values of Z."""
        @qml.qnode(self.dev, interface="torch")
        def qnode(x_input):
            self._full_ansatz(x_input, self.params)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        return qnode(x)

def create_autoencoder(config: AutoencoderConfig) -> AutoencoderModel:
    """Factory that returns a configured quantum autoencoder."""
    return AutoencoderModel(config)

def train_autoencoder(
    model: AutoencoderModel,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> List[float]:
    """Training loop for the quantum autoencoder."""
    device = device or torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x in data:
            x = x.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(data)
        history.append(epoch_loss)
    return history

__all__ = ["AutoencoderModel", "AutoencoderConfig", "create_autoencoder", "train_autoencoder"]
