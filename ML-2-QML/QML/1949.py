"""Quantum autoencoder using Pennylane and a variational circuit."""
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Callable, List, Optional, Iterable

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

class Autoencoder:
    """Variational quantum autoencoder implemented with Pennylane.
    The encoder is a parameterised circuit that maps classical data to
    a low‑dimensional quantum state.  The decoder is a classical linear
    layer that reconstructs the input from the expectation values of
    the latent qubits.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 depth: int = 3,
                 device: str = "default.qubit",
                 lr: float = 0.01,
                 optimizer: str = "Adam",
                 loss_fn: Callable = nn.MSELoss()):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.depth = depth
        self.dev = qml.device(device, wires=latent_dim)
        self.loss_fn = loss_fn
        self.lr = lr

        # Parameterised circuit
        self.params = nn.Parameter(torch.randn((self.latent_dim, self.depth)))
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")
        # Classical decoder
        self.decoder = nn.Linear(self.latent_dim, self.input_dim)

        # Optimiser
        opt_cls = getattr(torch.optim, optimizer)
        self.opt = opt_cls([self.params, *self.decoder.parameters()], lr=self.lr)

    def _circuit(self, x: torch.Tensor, params: torch.Tensor):
        """Encode classical data into a quantum state."""
        # Feature map
        qml.templates.AngleEmbedding(x, wires=range(self.latent_dim))
        # Variational layer
        qml.templates.StronglyEntanglingLayers(params, wires=range(self.latent_dim))
        # Latent representation: expectation of PauliZ on each latent qubit
        return [qml.expval(qml.PauliZ(w)) for w in range(self.latent_dim)]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnode(x, self.params)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def train_autoencoder(self,
                          data: torch.Tensor,
                          *,
                          epochs: int = 200,
                          batch_size: int = 64,
                          early_stopping: int = 10,
                          device: Optional[torch.device] = None) -> List[float]:
        """Train the variational autoencoder."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qnode.device = self.dev  # ensure correct device
        self.decoder.to(device)
        self.params.to(device)
        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        history: List[float] = []

        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            self.opt.zero_grad()
            self.train()
            for (batch,) in loader:
                batch = batch.to(device)
                recon = self(batch)
                loss = self.loss_fn(recon, batch)
                loss.backward()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            self.opt.step()
            history.append(epoch_loss)

            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if early_stopping > 0 and epochs_no_improve >= early_stopping:
                print(f"Early stopping at epoch {epoch+1}")
                break
        return history

    def evaluate(self, data: torch.Tensor) -> torch.Tensor:
        """Return reconstruction errors for the provided data."""
        self.eval()
        with torch.no_grad():
            recon = self.forward(_as_tensor(data).to(self.decoder.weight.device))
            return torch.mean((recon - _as_tensor(data).to(self.decoder.weight.device)) ** 2, dim=1)

__all__ = ["Autoencoder"]
