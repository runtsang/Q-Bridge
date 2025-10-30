"""Hybrid autoencoder – classical backbone + quantum regularizer.

The module exports two top‑level factories, ``QuantumAutoencoderHybrid`` and
``QuantumAutoencoderHybridConfig``.  The classical encoder/decoder is
identical to the original PyTorch autoencoder but now accepts a
``quantum_latent_dim`` that is fed into a Qiskit ``SamplerQNN``.  The
quantum part is built via a swap‑test circuit that measures the overlap
with a fixed reference state.  The training loop uses a combined loss:
reconstruction MSE + a quantum fidelity penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --------------------------------------------------------------------------- #
# Classical autoencoder – encoder / decoder
# --------------------------------------------------------------------------- #

@dataclass
class QuantumAutoencoderHybridConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # Qiskit parameters
    quantum_latent_dim: int = 3   # number of qubits in the latent circuit
    quantum_reps: int = 1         # repetitions of RealAmplitudes ansatz
    device: torch.device | None = None


class ClassicalEncoder(nn.Module):
    """Standard fully‑connected encoder."""
    def __init__(self, cfg: QuantumAutoencoderHybridConfig) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if cfg.dropout:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassicalDecoder(nn.Module):
    """Standard fully‑connected decoder."""
    def __init__(self, cfg: QuantumAutoencoderHybridConfig) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if cfg.dropout:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# --------------------------------------------------------------------------- #
# Quantum helper – swap‑test circuit + SamplerQNN
# --------------------------------------------------------------------------- #

def _build_swap_test_circuit(
    quantum_latent_dim: int,
    quantum_reps: int,
) -> "qiskit.circuit.QuantumCircuit":
    """Return a circuit that encodes parameters into a register and
    performs a swap‑test with a fixed reference state |0…0⟩.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import RealAmplitudes

    # Total qubits: ancilla + 2 * quantum_latent_dim
    ancilla = 0
    regA_start = 1
    regB_start = regA_start + quantum_latent_dim
    total_qubits = 1 + 2 * quantum_latent_dim

    qc = QuantumCircuit(total_qubits)

    # Prepare ancilla in |+⟩
    qc.h(ancilla)

    # Reference register B is already |0…0⟩
    # Encode latent parameters into register A via RealAmplitudes
    theta = ParameterVector("theta", quantum_latent_dim * quantum_reps)
    ansatz = RealAmplitudes(quantum_latent_dim, reps=quantum_reps)
    ansatz = ansatz.assign_parameters(theta)
    qc.compose(ansatz, range(regA_start, regA_start + quantum_latent_dim), inplace=True)

    # Controlled‑swap between A and B, controlled by ancilla
    for qubit in range(quantum_latent_dim):
        qc.cswap(ancilla, regA_start + qubit, regB_start + qubit)

    # Measure ancilla
    qc.h(ancilla)
    qc.measure(ancilla, 0)

    return qc


def _quantum_fidelity_interpret(x: np.ndarray) -> float:
    """Interpret the raw measurement probability of the ancilla
    into a fidelity estimate: P(0) = (1 + F)/2 ⇒ F = 2*P(0) - 1
    """
    return float(2 * x[0] - 1)


def build_quantum_autoencoder_qnn(
    quantum_latent_dim: int,
    quantum_reps: int,
) -> "qiskit_machine_learning.neural_networks.SamplerQNN":
    """Return a SamplerQNN that implements the swap‑test circuit.

    The circuit has `quantum_latent_dim * quantum_reps` free parameters
    that will be supplied by the latent vector during training.
    """
    from qiskit import Aer
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit_machine_learning.optimizers import COBYLA

    qc = _build_swap_test_circuit(quantum_latent_dim, quantum_reps)
    sampler = Aer.get_backend("qasm_simulator")
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=[p for p in qc.parameters],
        interpret=_quantum_fidelity_interpret,
        output_shape=1,
        sampler=sampler,
        optimizer=COBYLA(maxiter=50),
    )
    return qnn


# --------------------------------------------------------------------------- #
# Hybrid model – combines classical and quantum parts
# --------------------------------------------------------------------------- #

class QuantumAutoencoderHybrid(nn.Module):
    """Hybrid autoencoder that adds a quantum fidelity regularizer."""
    def __init__(self, cfg: QuantumAutoencoderHybridConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ClassicalEncoder(cfg)
        self.decoder = ClassicalDecoder(cfg)
        self.qnn = build_quantum_autoencoder_qnn(
            cfg.quantum_latent_dim, cfg.quantum_reps
        )
        self.device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def quantum_fidelity(self, z: torch.Tensor) -> torch.Tensor:
        """Compute fidelity of the latent vector via the quantum circuit."""
        # The circuit expects `quantum_latent_dim * quantum_reps` parameters.
        # Pad or truncate the latent vector accordingly.
        num_params = self.cfg.quantum_latent_dim * self.cfg.quantum_reps
        z_np = z.detach().cpu().numpy()
        if z_np.size > num_params:
            z_np = z_np[:num_params]
        elif z_np.size < num_params:
            z_np = np.concatenate([z_np, np.zeros(num_params - z_np.size)])
        # QNN expects a 2‑D array of shape (1, num_params)
        fidelity_val = self.qnn(z_np.reshape(1, -1))[0]
        return torch.tensor(fidelity_val, dtype=z.dtype, device=z.device)


def train_hybrid_autoencoder(
    model: QuantumAutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    lambda_fid: float = 0.1,
    device: torch.device | None = None,
) -> List[float]:
    """Train the hybrid autoencoder with a combined loss.

    Args:
        model: Hybrid model.
        data: Input data as a 2‑D tensor (N, D).
        epochs: Training epochs.
        batch_size: Mini‑batch size.
        lr: Learning rate.
        weight_decay: L2 penalty.
        lambda_fid: Weight of the quantum fidelity penalty.
        device: Optional device override.

    Returns:
        List of epoch‑level reconstruction losses.
    """
    device = device or model.device
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
            z = model.encoder(batch)
            recon = model.decoder(z)
            loss_rec = loss_fn(recon, batch)
            fidelity = model.quantum_fidelity(z)
            loss = loss_rec + lambda_fid * (1.0 - fidelity)
            loss.backward()
            optimizer.step()
            epoch_loss += loss_rec.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "QuantumAutoencoderHybrid",
    "QuantumAutoencoderHybridConfig",
    "train_hybrid_autoencoder",
]
