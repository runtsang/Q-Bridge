import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List, Optional

@dataclass
class AutoencoderGen48Config:
    input_dim: int
    latent_dim: int = 3
    num_qubits: int = 4
    num_layers: int = 2
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 32
    device: str = "default.qubit"

class AutoencoderGen48(nn.Module):
    def __init__(self, cfg: AutoencoderGen48Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.dev = qml.device(cfg.device, wires=cfg.num_qubits)
        self.qnode = qml.QNode(self._quantum_encoder, self.dev, interface="torch")
        self.decoder = nn.Linear(cfg.latent_dim, cfg.input_dim)
        self.optimizer = qml.AdamOptimizer(cfg.lr)

    def _quantum_encoder(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        qml.AmplitudeEmbedding(x, wires=range(self.cfg.num_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(self.cfg.num_qubits))
        return qml.state()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent)

    def train(self, data: torch.Tensor) -> List[float]:
        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        loss_history: List[float] = []
        for epoch in range(self.cfg.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.requires_grad_(True)
                def loss_fn(params):
                    latent = self._quantum_encoder(batch, params)
                    recon = self.decoder(latent)
                    return ((recon - batch)**2).mean()
                params, loss = self.optimizer.step_and_cost(loss_fn, self.qnode.weights)
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            loss_history.append(epoch_loss)
        return loss_history

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = ["AutoencoderGen48", "AutoencoderGen48Config"]
