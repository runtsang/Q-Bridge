import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli

def _as_tensor(data):
    """Convert input to a float32 tensor on the current device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class UnifiedAutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    quantum_n_qubits: int = 3
    quantum_layers: int = 2

class QuantumLatentBlock(nn.Module):
    """
    Quantum variational block that refines the classical latent vector.
    It encodes the vector via RY rotations, applies a RealAmplitudes ansatz,
    and returns the Z‑expectation values of each qubit.
    """
    def __init__(self, latent_dim: int, n_qubits: int, n_layers: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Trainable parameters for the ansatz
        self.ansatz_params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3, dtype=torch.float32)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: Tensor of shape (batch, latent_dim)
        Returns:
            Tensor of shape (batch, n_qubits) containing Z‑expectations.
        """
        batch = latent.shape[0]
        results = []
        for i in range(batch):
            vec = latent[i].cpu().numpy()
            qc = QuantumCircuit(self.n_qubits)
            # Encode the classical latent vector onto the first min(latent_dim, n_qubits) qubits
            for q in range(min(self.latent_dim, self.n_qubits)):
                theta = vec[q]
                qc.ry(theta, q)
            # RealAmplitudes ansatz
            for rep in range(self.n_layers):
                for q in range(self.n_qubits):
                    for gate_idx, gate in enumerate(['rx', 'ry', 'rz']):
                        param = self.ansatz_params[rep, q, gate_idx].item()
                        if gate == 'rx':
                            qc.rx(param, q)
                        elif gate == 'ry':
                            qc.ry(param, q)
                        else:
                            qc.rz(param, q)
            # Simulate
            state = Statevector.from_int(0, dims=(2**self.n_qubits,))
            state = state.evolve(qc)
            # Compute Z‑expectation for each qubit
            z_exp = []
            for q in range(self.n_qubits):
                pauli_z = Pauli('Z')
                exp_val = state.expectation_value(pauli_z, qubit=q)
                z_exp.append(exp_val)
            results.append(z_exp)
        return torch.tensor(np.array(results, dtype=np.float32), device=latent.device)

class UnifiedAutoencoderNet(nn.Module):
    """
    Hybrid autoencoder that combines a classical MLP encoder‑decoder
    with a quantum variational refinement of the latent representation.
    """
    def __init__(self, config: UnifiedAutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Classical encoder
        enc_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, hidden))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Quantum refinement block
        self.quantum_block = QuantumLatentBlock(
            latent_dim=config.latent_dim,
            n_qubits=config.quantum_n_qubits,
            n_layers=config.quantum_layers
        )

        # Classical decoder
        dec_layers = []
        in_dim = config.quantum_n_qubits  # quantum output dimension
        hidden_dims_rev = list(reversed(config.hidden_dims))
        for hidden in hidden_dims_rev:
            dec_layers.append(nn.Linear(in_dim, hidden))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        refined = self.quantum_block(latent)
        return self.decoder(refined)

def UnifiedAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum_n_qubits: int = 3,
    quantum_layers: int = 2,
) -> UnifiedAutoencoderNet:
    config = UnifiedAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        quantum_n_qubits=quantum_n_qubits,
        quantum_layers=quantum_layers,
    )
    return UnifiedAutoencoderNet(config)

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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "UnifiedAutoencoder",
    "UnifiedAutoencoderConfig",
    "UnifiedAutoencoderNet",
    "train_autoencoder",
]
