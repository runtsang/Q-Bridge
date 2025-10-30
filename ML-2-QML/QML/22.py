"""Quantum autoencoder using Pennylane variational circuits and a classical decoder.

Key features:
- Variational RealAmplitudes encoding circuit with angle embedding of input data.
- Classical decoder network to reconstruct the input from the latent vector.
- Fidelity evaluation between input state and reconstructed state.
- Domain‑wall qubit manipulation for testing purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Callable

import pennylane as qml
import pennylane.numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    """Configuration for the quantum autoencoder."""
    n_qubits: int
    latent_dim: int = 3
    hidden_dims: Tuple[int, int] = (64, 32)
    learning_rate: float = 1e-3
    epochs: int = 200
    batch_size: int = 32
    device: str = "default.qubit"
    seed: int = 42


# --------------------------------------------------------------------------- #
# Helper: domain‑wall qubit flip
# --------------------------------------------------------------------------- #

def domain_wall(circuit: qml.QNode, start: int, end: int) -> qml.QNode:
    """Return a new QNode inserting X gates from start to end-1 indices."""
    def wrapped(*args, **kwargs):
        circuit(*args, **kwargs)
        for i in range(start, end):
            qml.X(i)
    return wrapped


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class Autoencoder(nn.Module):
    """
    Hybrid quantum‑classical autoencoder.

    The encoder is a variational circuit built on top of an angle‑embedding
    layer that maps classical data to a quantum state.  The decoder is a
    lightweight classical MLP that reconstructs the original input from
    the latent vector produced by the circuit.
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        qml.set_options(device=config.device, shots=1024, seed=config.seed)

        # ----- Encoder -----
        self.encoder = qml.QNode(
            self._encoder_circuit,
            dev=qml.device(config.device, wires=config.n_qubits),
            interface="torch",
            diff_method="backprop",
        )

        # ----- Decoder -----
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in config.hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.n_qubits))
        self.decoder = nn.Sequential(*decoder_layers)

    # --------------------------------------------------------------------- #
    # Encoder circuit
    # --------------------------------------------------------------------- #

    def _encoder_circuit(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Variational circuit that takes the input vector x (angle embedding)
        and a weight vector for the RealAmplitudes ansatz.
        Returns the expectation values of PauliZ on each qubit.
        """
        # Angle embedding
        for i, val in enumerate(x):
            qml.PhaseShift(val, wires=i)
        # Variational layer
        qml.templates.RealAmplitudes(
            weights,
            wires=range(self.config.n_qubits),
            reps=3,
        )
        return [qml.expval(qml.PauliZ(i)) for i in range(self.config.n_qubits)]

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input using the quantum circuit and decode classically.
        """
        # Ensure input shape (batch, n_qubits)
        batch_size = x.shape[0]
        # Random initial weights for the ansatz
        weights = torch.randn(self.config.n_qubits * 3 * 3, requires_grad=True, device=x.device)
        # Encode
        latent = self.encoder(x, weights)  # shape (batch, n_qubits)
        # Decode
        reconstruction = self.decoder(latent)
        return reconstruction

    # --------------------------------------------------------------------- #
    # Training helper
    # --------------------------------------------------------------------- #

    def train_autoencoder(
        self,
        data: torch.Tensor,
        *,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        device: torch.device | None = None,
    ) -> List[float]:
        """
        Train the hybrid autoencoder using Adam on both quantum and classical parameters.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        data = _as_tensor(data).to(device)

        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        learning_rate = learning_rate or self.config.learning_rate

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        history: List[float] = []

        loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon = self(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(loader.dataset)
            history.append(epoch_loss)
            if epoch % 20 == 0:
                print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")
        return history

    # --------------------------------------------------------------------- #
    # Fidelity evaluation
    # --------------------------------------------------------------------- #

    def fidelity(self, x: torch.Tensor) -> float:
        """
        Compute the average fidelity between the input state and the reconstructed state
        using a state‑vector simulator.
        """
        dev = qml.device("default.qubit", wires=self.config.n_qubits)
        @qml.qnode(dev, interface="torch")
        def state_vector_circuit(x_in: torch.Tensor) -> torch.Tensor:
            for i, val in enumerate(x_in):
                qml.PhaseShift(val, wires=i)
            return qml.state()

        # Input state
        input_state = state_vector_circuit(x)
        # Reconstructed state via decoder output
        latent = self.encoder(x, torch.randn_like(self.encoder.weights))
        recon_state = self.decoder(latent)
        # Compute fidelity
        fidelity = torch.abs(torch.dot(input_state, recon_state.conj())) ** 2
        return fidelity.item()

    # --------------------------------------------------------------------- #
    # Utility: domain‑wall qubit manipulation
    # --------------------------------------------------------------------- #

    def apply_domain_wall(self, start: int, end: int) -> None:
        """
        Apply an X gate to all qubits in the range [start, end).
        This modifies the encoder circuit in place for testing.
        """
        orig_circuit = self.encoder
        self.encoder = domain_wall(orig_circuit, start, end)


# --------------------------------------------------------------------------- #
# Helper: tensor conversion (mirrors the classical seed)
# --------------------------------------------------------------------------- #

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


__all__ = ["Autoencoder", "AutoencoderConfig"]
