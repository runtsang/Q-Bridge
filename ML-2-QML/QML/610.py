from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Sequence

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class QuantumAutoencoderConfig:
    input_dim: int
    latent_dim: int = 2
    hidden_layers: Tuple[int,...] = (32, 16)
    num_qubits: int = 4
    backend: str = "default.qubit"
    device: Optional[qml.Device] = None
    weight_init: str = "normal"

class Autoencoder(nn.Module):
    """Hybrid quantum‑classical autoencoder using PennyLane."""

    def __init__(self, cfg: QuantumAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_qubits = max(cfg.input_dim, cfg.latent_dim)
        self.dev = cfg.device or qml.device(cfg.backend, wires=self.num_qubits)

        # Classical decoder
        layers: list[nn.Module] = []
        prev = cfg.latent_dim
        for h in cfg.hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, cfg.input_dim))
        self.decoder = nn.Sequential(*layers)

        # Parameters for the ansatz
        self.ansatz_params = nn.Parameter(
            torch.randn(self.num_qubits * self.num_qubits, requires_grad=True)
        )

        # Quantum encoder as a QNode
        def circuit(x, params):
            # Feature map
            for i in range(cfg.input_dim):
                qml.RY(x[i], wires=i)
            # Ansatz
            idx = 0
            for i in range(self.num_qubits):
                for j in range(self.num_qubits):
                    qml.CNOT(wires=[i, j])
                    qml.RZ(params[idx], wires=j)
                    idx += 1
            # Readout on the first latent_dim qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(cfg.latent_dim)]

        self.qnode = qml.QNode(circuit, self.dev, interface="torch", diff_method="parameter-shift")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_out = []
        for sample in x:
            out = self.qnode(sample, self.ansatz_params)
            batch_out.append(out)
        return torch.stack(batch_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decoder(z)

def train_autoencoder(
    model: Autoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
        if verbose:
            print(f"Epoch {epoch+1:03d} | Loss {epoch_loss:.4f}")
    return history

__all__ = ["Autoencoder", "QuantumAutoencoderConfig", "train_autoencoder"]
