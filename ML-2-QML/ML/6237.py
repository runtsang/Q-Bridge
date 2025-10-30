import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import MSELoss
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

import optuna
import pennylane as qml
from pennylane import numpy as pnp

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid classical‑quantum autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # Quantum part
    qubits: int = 8
    depth: int = 3
    quantum_weight: float = 0.01   # weight of the quantum regulariser

# --------------------------------------------------------------------------- #
# Quantum regulariser
# --------------------------------------------------------------------------- #
class QuantumRegulariser(nn.Module):
    """
    Parameterised quantum circuit that acts on the latent vector.
    The circuit is trained jointly with the classical network.
    """
    def __init__(self, latent_dim: int, qubits: int, depth: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.qubits = qubits
        self.depth = depth

        # Trainable parameters of the circuit
        self.params = nn.Parameter(torch.randn(qubits * depth * 3))

        # Pennylane QNode
        dev = qml.device("default.qubit", wires=qubits)
        self.qnode = qml.QNode(self._circuit, dev, interface="torch")

    def _circuit(self, latent: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        # Encode the latent vector into qubit rotations
        for i in range(min(self.latent_dim, self.qubits)):
            qml.RX(latent[i], wires=i)

        # Parameterised layers
        p_idx = 0
        for _ in range(self.depth):
            for w in range(self.qubits):
                qml.RY(params[p_idx], wires=w); p_idx += 1
                qml.RZ(params[p_idx], wires=w); p_idx += 1
                qml.CNOT(wires=[w, (w + 1) % self.qubits]); p_idx += 1

        # Expectation value of Z on first qubit
        return qml.expval(qml.PauliZ(0))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.qnode(latent, *self.params)

# --------------------------------------------------------------------------- #
# Classical autoencoder
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """
    Hybrid autoencoder: classical MLP encoder/decoder + quantum latent regulariser.
    """
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config

        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Quantum regulariser
        self.qreg = QuantumRegulariser(
            latent_dim=config.latent_dim,
            qubits=config.qubits,
            depth=config.depth
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def quantum_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Return mean‑squared error of the quantum circuit output to 0."""
        out = self.qreg(z)
        return ((out - 0.0) ** 2).mean()

# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    early_stop_patience: int = 10,
    verbose: bool = False,
) -> Tuple[List[float], List[float]]:
    """
    Train the hybrid autoencoder. Returns two lists: classical loss history and quantum loss history.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    loss_fn = MSELoss()

    best_loss = float("inf")
    patience_counter = 0
    cls_history: List[float] = []
    q_history: List[float] = []

    for epoch in range(epochs):
        epoch_cls_loss = 0.0
        epoch_q_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            cls_loss = loss_fn(recon, batch)

            z = model.encode(batch)
            q_loss = model.quantum_loss(z)

            loss = cls_loss + model.config.quantum_weight * q_loss
            loss.backward()
            optimizer.step()

            epoch_cls_loss += cls_loss.item() * batch.size(0)
            epoch_q_loss += q_loss.item() * batch.size(0)

        epoch_cls_loss /= len(dataset)
        epoch_q_loss /= len(dataset)
        cls_history.append(epoch_cls_loss)
        q_history.append(epoch_q_loss)

        if verbose:
            print(f"Epoch {epoch+1:3d} | cls:{epoch_cls_loss:.4f} | q:{epoch_q_loss:.4f}")

        # Early stopping
        if epoch_cls_loss < best_loss:
            best_loss = epoch_cls_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose:
                    print("Early stopping triggered.")
                break

        scheduler.step()

    return cls_history, q_history


def _as_tensor(data: torch.Tensor | np.ndarray | list) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Hyper‑parameter tuner (Optuna)
# --------------------------------------------------------------------------- #
def tune_quantum_params(
    data: torch.Tensor,
    config: AutoencoderConfig,
    *,
    n_trials: int = 30,
    epochs: int = 30,
    verbose: bool = False,
) -> AutoencoderConfig:
    """
    Use Optuna to optimise the number of qubits and the depth of the quantum circuit.
    """
    def objective(trial):
        qubits = trial.suggest_int("qubits", 4, 12)
        depth = trial.suggest_int("depth", 1, 5)
        trial_config = AutoencoderConfig(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            qubits=qubits,
            depth=depth,
            quantum_weight=config.quantum_weight
        )
        model = AutoencoderNet(trial_config)
        cls_hist, q_hist = train_autoencoder(
            model, data, epochs=epochs, verbose=False
        )
        return cls_hist[-1] + trial_config.quantum_weight * q_hist[-1]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    return AutoencoderConfig(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        qubits=best_params["qubits"],
        depth=best_params["depth"],
        quantum_weight=config.quantum_weight
    )

# --------------------------------------------------------------------------- #
# Public factory
# --------------------------------------------------------------------------- #
def autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    qubits: int = 8,
    depth: int = 3,
    quantum_weight: float = 0.01,
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        qubits=qubits,
        depth=depth,
        quantum_weight=quantum_weight,
    )
    return AutoencoderNet(config)


__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
    "tune_quantum_params",
    "autoencoder",
]
