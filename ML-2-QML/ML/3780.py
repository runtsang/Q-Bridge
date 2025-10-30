"""Hybrid classical encoder for AutoencoderGen075.

The module defines a classical neural network that encodes high‑dimensional inputs into a latent space, and a quantum helper that takes the latent vector as input and returns a variational circuit representation for reconstruction. The two halves are connected through a joint loss that combines MSE on the latent vectors with a fidelity term computed from the quantum decoder. This design mirrors the classic Autoencoder example while adding a quantum‑friendly interface.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Helper: float‑tensor conversion
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current device."""
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# Configuration dataclass
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for a hybrid AutoencoderGen‑075."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    fidelity_weight: float = 0.5   # weight for quantum fidelity loss

# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """A classical MLP auto‑encoder (the *encoder* part)."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        # Latent layer
        self.latent = nn.Linear(in_dim, config.latent_dim)
        # Decoder (classical reconstruction)
        decoder_layers: List[nn.Module] = [
            nn.Linear(config.latent_dim, config.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[-1], config.input_dim)
        ]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.latent(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# --------------------------------------------------------------------------- #
class QuantumDecoder:
    """Wrapper for the quantum decoder circuit used in the hybrid model.

    The class exposes a ``reconstruct`` method that takes a latent vector
    and returns a classical vector by sampling the quantum circuit.
    It uses a Qiskit ``Sampler`` and a ``RealAmplitudes`` ansatz.
    """
    def __init__(self, latent_dim: int, trash_dim: int = 2, reps: int = 5, device: Optional[str] = None) -> None:
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        from qiskit.circuit.library import RealAmplitudes
        from qiskit.primitives import Sampler

        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self.device = device or 'qasm_simulator'

        # Build the circuit
        qr = QuantumRegister(latent_dim + 2 * trash_dim + 1, 'q')
        cr = ClassicalRegister(1, 'c')
        self.circuit = QuantumCircuit(qr, cr)
        # Ansatz on latent + first trash qubits
        self.circuit.append(RealAmplitudes(latent_dim + trash_dim, reps=reps), range(0, latent_dim + trash_dim))
        self.circuit.barrier()
        # Swap test between latent and trash
        aux = latent_dim + 2 * trash_dim
        self.circuit.h(aux)
        for i in range(trash_dim):
            self.circuit.cswap(aux, latent_dim + i, latent_dim + trash_dim + i)
        self.circuit.h(aux)
        # Measurement
        self.circuit.measure(aux, cr[0])

        self.sampler = Sampler()

    def reconstruct(self, z: np.ndarray | torch.Tensor) -> np.ndarray:
        """Given a latent vector, encode it as a computational basis state and
        sample the circuit to obtain a probability distribution over the
        auxiliary qubit.  The returned vector is the expectation value of
        the auxiliary measurement, reshaped to the latent dimension.
        """
        import numpy as np
        # Convert to numpy array
        if isinstance(z, torch.Tensor):
            z_np = z.detach().cpu().numpy()
        else:
            z_np = np.asarray(z)
        # Encode latent as computational basis state by thresholding
        basis = (np.round(z_np) % 2).astype(int)
        # Prepare the parameter bindings (all zeros for simplicity)
        param_binds = [{p: 0.0 for p in self.circuit.parameters}]
        # Run sampler
        result = self.sampler.run(self.circuit, shots=1024, parameter_binds=param_binds)
        counts = result.get_counts()
        # Compute expectation of auxiliary qubit
        exp = 0.0
        for out, n in counts.items():
            exp += (1 if out == '1' else -1) * n
        exp /= 1024
        # Map expectation to a vector of shape latent_dim
        return np.full(self.latent_dim, exp)

    def fidelity(self, target: np.ndarray, output: np.ndarray) -> float:
        """Quantum fidelity between target and output vectors (treated as states)."""
        a = target / (np.linalg.norm(target) + 1e-12)
        b = output / (np.linalg.norm(output) + 1e-12)
        return float(np.abs(np.vdot(a, b)) ** 2)

# --------------------------------------------------------------------------- #
class AutoencoderGen075:
    """Hybrid auto‑encoder that couples a classical encoder with a quantum decoder."""
    def __init__(self, config: AutoencoderConfig, quantum_decoder: QuantumDecoder) -> None:
        self.model = AutoencoderNet(config)
        self.quantum = quantum_decoder
        self.fidelity_weight = config.fidelity_weight

    def train(self,
              data: torch.Tensor,
              epochs: int = 50,
              batch_size: int = 64,
              lr: float = 1e-3) -> List[float]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        dataloader = DataLoader(TensorDataset(_as_tensor(data)), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch, in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                z = self.model.encode(batch)
                # Classical reconstruction loss
                recon = self.model.decode(z)
                loss = loss_fn(recon, batch)
                # Quantum fidelity loss per sample in the batch
                for i in range(z.size(0)):
                    z_np_single = z[i].detach().cpu().numpy()
                    q_out = self.quantum.reconstruct(z_np_single)
                    fid = self.quantum.fidelity(z_np_single, q_out)
                    loss += self.fidelity_weight * (1 - fid)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataloader.dataset)
            history.append(epoch_loss)
            print(f'Epoch {epoch+1}/{epochs} loss={epoch_loss:.4f}')
        return history

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z)

    def reconstruct(self, z: torch.Tensor) -> np.ndarray:
        return self.quantum.reconstruct(z.detach().cpu().numpy())

__all__ = ['AutoencoderConfig', 'AutoencoderNet', 'QuantumDecoder', 'AutoencoderGen075']
