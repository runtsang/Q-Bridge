"""Hybrid variational autoencoder combining classical encoding with a Qiskit quantum feature map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, PauliFeatureMap
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    residual: bool = False
    layer_norm: bool = False


class _LinearBlock(nn.Module):
    """Linear → (optional residual) → ReLU → (optional dropout) → (optional layer‑norm)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float, norm: bool, residual: bool):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.residual = residual and in_dim == out_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm = nn.LayerNorm(out_dim) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.residual:
            out = out + x
        out = self.relu(out)
        out = self.dropout(out)
        out = self.norm(out)
        return out


class HybridAutoencoder(nn.Module):
    """Hybrid classical‑quantum autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        # Classical encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(_LinearBlock(in_dim, hidden, config.dropout, config.layer_norm, config.residual))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.classical_encoder = nn.Sequential(*encoder_layers)

        # Quantum feature map + ansatz
        self.num_qubits = config.latent_dim
        self.feature_map = PauliFeatureMap(feature_dimension=self.num_qubits, reps=1, paulis="XZ")
        self.ansatz = RealAmplitudes(self.num_qubits, reps=2)

        # Combine into a single circuit
        self.circuit = QuantumCircuit(self.num_qubits)
        self.circuit.append(self.feature_map, range(self.num_qubits))
        self.circuit.append(self.ansatz, range(self.num_qubits))

        # Sampler and QNN
        sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            interpret=self._expectation_z,
            output_shape=(self.num_qubits,),
            sampler=sampler,
        )

        # Classical decoder
        decoder_layers = []
        in_dim = self.num_qubits
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(_LinearBlock(in_dim, hidden, config.dropout, config.layer_norm, config.residual))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.classical_decoder = nn.Sequential(*decoder_layers)

    def _expectation_z(self, statevector: torch.Tensor) -> torch.Tensor:
        """Return expectation values of Z on each qubit."""
        # statevector shape: (2**n,)
        probs = statevector.abs() ** 2
        probs = probs.reshape(-1, 2 ** self.num_qubits)
        # Compute expectation value for each qubit
        expectations = []
        for qubit in range(self.num_qubits):
            # Build projector for |0> and |1>
            mask0 = 1 << (self.num_qubits - qubit - 1)
            mask1 = ~mask0
            exp = (probs[:, mask0] - probs[:, mask1]).sum()
            expectations.append(exp)
        return torch.stack(expectations, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical encoding
        latent = self.classical_encoder(x)

        # Quantum encoding
        # The QNN expects a 1‑D tensor per sample; we flatten latent
        q_input = latent.view(-1, self.num_qubits)
        q_output = self.qnn(q_input)

        # Classical decoding
        reconstruction = self.classical_decoder(q_output)
        return reconstruction


def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    residual: bool = False,
    layer_norm: bool = False,
) -> HybridAutoencoder:
    """Factory returning a configured hybrid autoencoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        residual=residual,
        layer_norm=layer_norm,
    )
    return HybridAutoencoder(config)


def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop for the hybrid autoencoder."""
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
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["HybridAutoencoder", "HybridAutoencoderFactory", "train_hybrid_autoencoder"]
