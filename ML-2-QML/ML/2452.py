import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable
from quantum_autoencoder import QuantumEncoderCircuit

@dataclass
class UnifiedAutoencoderConfig:
    """Configuration for the fused autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # quantum‑friendly hyper‑parameters
    q_latent_dim: int = 4          # number of qubits for the quantum part
    q_reps: int = 3                # repeat count for the ansatz
    q_shift: float = 0.0           # shift for parameter‑shift rule

class UnifiedAutoencoderNet(nn.Module):
    """Hybrid autoencoder that couples a classical encoder/decoder with a
    quantum variational circuit.  The quantum circuit is used as a latent
    space regularizer: it encodes the latent vector into a quantum state,
    and the expectation value of a Pauli‑Z measurement is used as an
    additional reconstruction loss term.
    """
    def __init__(self, config: UnifiedAutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._build_mlp(
            config.input_dim, config.hidden_dims, config.latent_dim, config.dropout
        )
        self.quantum_circuit = QuantumEncoderCircuit(
            num_qubits=config.q_latent_dim, reps=config.q_reps
        )
        self.decoder = self._build_mlp(
            config.latent_dim,
            list(reversed(config.hidden_dims)),
            config.input_dim,
            config.dropout,
        )

    def _build_mlp(
        self, in_dim: int, hidden_dims: Tuple[int,...], out_dim: int, dropout: float
    ) -> nn.Sequential:
        layers = []
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        # Use only the first `q_latent_dim` components for the quantum circuit
        q_inputs = z[:, : self.quantum_circuit.num_qubits]
        q_exp = self.quantum_circuit.expectation(q_inputs)
        return self.decode(z), q_exp

def train_autoencoder(
    model: UnifiedAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, q_exp = model(batch)
            loss = loss_fn(recon, batch)
            # add quantum regularizer
            loss += 0.01 * q_exp.mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = ["UnifiedAutoencoderNet", "UnifiedAutoencoderConfig", "train_autoencoder"]
