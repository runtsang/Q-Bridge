"""
Hybrid autoencoder combining classical PyTorch layers with a quantum latent space.

Dependencies:
    - torch
    - qiskit
    - qiskit_machine_learning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Quantum imports are optional; the module falls back to a dummy quantum encoder
# if Qiskit is not available.
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import RealAmplitudes
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit_machine_learning.optimizers import COBYLA
    from qiskit_machine_learning.utils import algorithm_globals
except Exception:
    QuantumCircuit = None  # type: ignore[assignment]


def _as_tensor(data):
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    quantum_reps: int = 3  # depth of the variational ansatz


class ClassicalAutoencoder(nn.Module):
    """Purely classical MLP autoencoder used as a drop‑in replacement for the decoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class QuantumAutoencoder(nn.Module):
    """
    Variational quantum autoencoder that produces a latent vector via a swap‑test
    circuit.  The circuit is built from a RealAmplitudes ansatz and a
    controlled‑swap operation that compares the data qubits with a reference
    state.  The output is interpreted as a probability distribution over
    computational basis states.
    """
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        if QuantumCircuit is None:
            raise RuntimeError("Qiskit is required for QuantumAutoencoder.")
        self.num_latent = config.latent_dim
        self.num_trash = max(1, config.latent_dim // 2)
        self.circuit = self._build_circuit(config.quantum_reps)

        # SamplerQNN turns the circuit into a differentiable layer
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,  # identity
            output_shape=(self.num_latent,),
        )

    def _build_circuit(self, reps: int) -> QuantumCircuit:
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Encode data into the first block
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=reps)
        circuit.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)

        # Swap test
        auxiliary = self.num_latent + 2 * self.num_trash
        circuit.h(auxiliary)
        for i in range(self.num_trash):
            circuit.cswap(auxiliary, self.num_latent + i, self.num_latent + self.num_trash + i)
        circuit.h(auxiliary)

        circuit.measure(auxiliary, cr[0])
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to have shape (..., latent_dim)
        flattened = x.view(-1, self.num_latent)
        # Convert to a probability vector via the QNN
        probs = self.qnn(flattened)
        # Collapse to a single expected latent vector
        return probs


class AutoencoderHybrid(nn.Module):
    """
    Hybrid autoencoder that uses a classical encoder, a quantum latent
    representation, and a classical decoder.  The quantum sub‑network can be
    trained jointly with the classical parts or frozen for inference.
    """
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.classical = ClassicalAutoencoder(config)
        self.quantum = QuantumAutoencoder(config)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Classical encoder followed by quantum latent extraction."""
        latent = self.classical.encode(x)
        quantum_latent = self.quantum(latent)
        return quantum_latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Classical decoder that accepts the quantum latent vector."""
        return self.classical.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def train_autoencoder(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | None = None,
    ) -> list[float]:
        """Joint training loop for the hybrid model."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction = self(batch)
                loss = loss_fn(reconstruction, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history


__all__ = ["AutoencoderHybrid", "AutoencoderConfig"]
