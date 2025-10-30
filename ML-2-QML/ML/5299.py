"""
Hybrid LSTM and auxiliary models combining classical and quantum backbones.

The module implements:
* `HybridQLSTM` – an LSTM cell that can optionally replace its gates with
  quantum variational circuits (provided by the QML module).
* `HybridLSTMTagger` – a sequence‑tagging wrapper that uses `HybridQLSTM`.
* `RegressionDataset` & `ClassicalRegressionModel` – the classical regression
  counterpart of the quantum regression example.
* `HybridQModel` – a regression head that delegates to a quantum module when
  available.
* `QFCModel` – a lightweight CNN → FC pipeline inspired by Quantum‑NAT.
* `AutoencoderNet` and `train_autoencoder` – a fully‑connected autoencoder
  for pre‑training or dimensionality reduction.

The design keeps the classical implementation self‑contained, while
providing hooks (`use_quantum`) that can be enabled when the quantum
dependencies (`torchquantum`) are available.  This allows the same code
path to be executed on CPU/GPU or on a quantum simulator without code
changes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, List, Optional

# --------------------------------------------------------------------------- #
# 1. Classical regression utilities (from ReferencePair[2])
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for regression."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset yielding feature vectors and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class ClassicalRegressionModel(nn.Module):
    """Simple feed‑forward regressor."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x.to(torch.float32)).squeeze(-1)

# --------------------------------------------------------------------------- #
# 2. Hybrid LSTM cell and tagging model (from ReferencePair[1])
# --------------------------------------------------------------------------- #

class HybridQLSTM(nn.Module):
    """
    LSTM cell that can optionally replace its linear gates with quantum
    variational circuits (provided by the QML module).  When `use_quantum`
    is False, the cell reduces to a standard classical LSTM cell.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = 0, use_quantum: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum and n_qubits > 0

        # Linear projections for the four gates
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits if self.use_quantum else hidden_dim)
        self.input_lin  = nn.Linear(input_dim + hidden_dim, n_qubits if self.use_quantum else hidden_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits if self.use_quantum else hidden_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits if self.use_quantum else hidden_dim)

        # Optional quantum layers (imported lazily to keep the module pure classical)
        if self.use_quantum:
            try:
                from.qml_code import QGateLayer
                self.forget_q = QGateLayer(n_qubits)
                self.input_q  = QGateLayer(n_qubits)
                self.update_q = QGateLayer(n_qubits)
                self.output_q = QGateLayer(n_qubits)
            except Exception as e:
                raise RuntimeError(
                    "Quantum layers requested but torchquantum is not installed."
                ) from e

    def forward(self, inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(
                self.forget_q(self.forget_lin(combined))
                if self.use_quantum else self.forget_lin(combined)
            )
            i = torch.sigmoid(
                self.input_q(self.input_lin(combined))
                if self.use_quantum else self.input_lin(combined)
            )
            g = torch.tanh(
                self.update_q(self.update_lin(combined))
                if self.use_quantum else self.update_lin(combined)
            )
            o = torch.sigmoid(
                self.output_q(self.output_lin(combined))
                if self.use_quantum else self.output_lin(combined)
            )
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class HybridLSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical and hybrid
    (quantum‑enhanced) LSTM cells.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, use_quantum: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim,
                                n_qubits=n_qubits,
                                use_quantum=use_quantum)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

# --------------------------------------------------------------------------- #
# 3. Hybrid regression head (combining classical and quantum)
# --------------------------------------------------------------------------- #

class HybridQModel(nn.Module):
    """
    Regression head that delegates to a quantum module when available.
    The quantum module is expected to be defined in the QML module and
    expose a `forward` method that accepts a batch of quantum states.
    """
    def __init__(self, num_wires: int, use_quantum: bool = False):
        super().__init__()
        self.use_quantum = use_quantum and num_wires > 0
        if self.use_quantum:
            try:
                from.qml_code import QRegressionLayer
                self.quantum_head = QRegressionLayer(num_wires)
            except Exception as e:
                raise RuntimeError(
                    "Quantum regression head requested but torchquantum is not installed."
                ) from e
        else:
            self.quantum_head = None
        # Classical linear head for fallback
        self.linear_head = nn.Linear(num_wires if self.use_quantum else 1, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        if self.use_quantum and self.quantum_head is not None:
            return self.quantum_head(state_batch)
        # Fallback: treat state_batch as a vector
        return self.linear_head(state_batch).squeeze(-1)

# --------------------------------------------------------------------------- #
# 4. Classical QFCModel (from ReferencePair[3])
# --------------------------------------------------------------------------- #

class QFCModel(nn.Module):
    """Simple CNN followed by a fully connected head."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

# --------------------------------------------------------------------------- #
# 5. Autoencoder utilities (from ReferencePair[4])
# --------------------------------------------------------------------------- #

from dataclasses import dataclass

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder."""
    def __init__(self, config: AutoencoderConfig):
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

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory for a configured autoencoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Simple reconstruction training loop."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

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

# --------------------------------------------------------------------------- #
# 6. Utility helpers
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

__all__ = [
    "HybridQLSTM",
    "HybridLSTMTagger",
    "RegressionDataset",
    "ClassicalRegressionModel",
    "HybridQModel",
    "QFCModel",
    "Autoencoder",
    "AutoencoderNet",
    "train_autoencoder",
    "AutoencoderConfig",
]
