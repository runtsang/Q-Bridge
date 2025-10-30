"""Hybrid autoencoder that unifies classical MLP and quantum variational encoder/decoder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
#  CONFIGURATION
# --------------------------------------------------------------------------- #
@dataclass
class UnifiedAutoEncoderConfig:
    """Configuration for the hybrid encoder/decoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # Quantum‑specific knobs
    qiskit_backend: str | None = None          # e.g. "qasm_simulator"
    qiskit_shots: int = 1024
    qiskit_seed: int = 42
    # Training hyper‑parameters
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: torch.device | None = None


# --------------------------------------------------------------------------- #
#  CLASSICAL CORE
# --------------------------------------------------------------------------- #
class ClassicalAutoEncoderNet(nn.Module):
    """Fully‑connected autoencoder that mirrors the original seed."""
    def __init__(self, cfg: UnifiedAutoEncoderConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class ClassicalAutoEncoder:
    """Convenience wrapper that exposes the same training API as the seed."""
    def __init__(self, cfg: UnifiedAutoEncoderConfig) -> None:
        self.cfg = cfg
        self.model = ClassicalAutoEncoderNet(cfg)
        self.device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, data: torch.Tensor, epochs: int | None = None) -> List[float]:
        epochs = epochs or self.cfg.epochs
        dataset = TensorDataset(self._as_tensor(data))
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        loss_fn = nn.MSELoss()
        history: List[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                recon = self.model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

    @staticmethod
    def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(data, dtype=torch.float32)
        if t.dtype!= torch.float32:
            t = t.to(dtype=torch.float32)
        return t


# --------------------------------------------------------------------------- #
#  QUANTUM ENCODER / DECOUPLER
# --------------------------------------------------------------------------- #
def _quantum_encoder_circuit(latent_dim: int, trash_dim: int) -> "qiskit.circuit.QuantumCircuit":
    """Build a Qiskit circuit that maps a classical vector to a latent state."""
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import RealAmplitudes

    qr = QuantumRegister(latent_dim + 2 * trash_dim + 1, "q")
    cr = ClassicalRegister(1, "c")
    circ = QuantumCircuit(qr, cr)

    # Feature mapping via a parameterised ansatz
    circ.compose(RealAmplitudes(latent_dim + trash_dim, reps=5), range(0, latent_dim + trash_dim), inplace=True)
    circ.barrier()

    # Swap‑test style entanglement with an ancilla
    ancilla = latent_dim + 2 * trash_dim
    circ.h(ancilla)
    for i in range(trash_dim):
        circ.cswap(ancilla, latent_dim + i, latent_dim + trash_dim + i)
    circ.h(ancilla)
    circ.measure(ancilla, cr[0])
    return circ


class QuantumDecoderNet(nn.Module):
    """Decoder that maps a quantum latent state back to a classical vector using a sampler."""
    def __init__(self, latent_dim: int, backend: str | None, shots: int, seed: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.backend = backend
        self.shots = shots
        self.seed = seed

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Convert latent vector to a probability distribution for sampling
        import qiskit
        from qiskit.primitives import Sampler as QiskitSampler
        from qiskit.quantum_info import Statevector

        # Build a simple measurement circuit that prepares the latent amplitudes
        circ = qiskit.QuantumCircuit(self.latent_dim)
        # Normalize the vector to a valid statevector
        vec = latent.cpu().numpy()
        vec = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
        circ.initialize(vec, range(self.latent_dim))
        sampler = QiskitSampler(backend=self.backend, shots=self.shots, seed_simulator=self.seed)
        result = sampler.run(circ).result()
        counts = result.get_counts()
        # Convert measurement counts to a probability distribution
        probs = np.array([counts.get(bit, 0) for bit in sorted(counts)]) / self.shots
        return torch.tensor(probs, dtype=torch.float32).to(latent.device)


# --------------------------------------------------------------------------- #
#  HYPER‑PARAMETER‑TUNING & TRAINING LOOP
# --------------------------------------------------------------------------- #
def train_unified_autoencoder(
    cfg: UnifiedAutoEncoderConfig,
    data: torch.Tensor,
) -> List[float]:
    """Full hybrid training loop that first trains the classical encoder/decoder
    and then fine‑tunes a quantum decoder (placeholder for a variational circuit)."""
    # 1. Classical training
    classical = ClassicalAutoEncoder(cfg)
    class_hist = classical.train(data)

    # 2. Quantum decoding step (placeholder for a variational circuit)
    #   In practice the quantum circuit would be defined by the quantum encoder
    #   and the latent space would be the output of the reconstructions.
    quantum_qnn = _quantum_encoder_circuit(
        cfg.latent_dim,
        trash_dim=cfg.hidden_dims[0] // 2,
    )
    # (No actual training of the quantum part in this simplified example)
    return class_hist


# --------------------------------------------------------------------------- #
#  PUBLIC API
# --------------------------------------------------------------------------- #
def UnifiedAutoEncoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    qiskit_backend: str | None = None,
    qiskit_shots: int = 1024,
    qiskit_seed: int = 42,
) -> ClassicalAutoEncoder:
    """Factory that returns a classical autoencoder with optional quantum support."""
    cfg = UnifiedAutoEncoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        qiskit_backend=qiskit_backend,
        qiskit_shots=qiskit_shots,
        qiskit_seed=qiskit_seed,
    )
    return ClassicalAutoEncoder(cfg)


__all__ = [
    "UnifiedAutoEncoder",
    "train_unified_autoencoder",
    "ClassicalAutoEncoder",
    "ClassicalAutoEncoderNet",
    "UnifiedAutoEncoderConfig",
]
