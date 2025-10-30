"""Quantum‑enhanced hybrid autoencoder using EstimatorQNN for latent refinement.

This module implements the same :class:`HybridAutoEncoder` as the classical
module, but replaces the optional refinement layer with a variational
quantum circuit.  The quantum layer is built with Qiskit’s
`RealAmplitudes` ansatz and evaluated through an `EstimatorQNN`.  A single
parameter‑free swap‑test inspired fidelity objective is used to refine the
latent representation.  The design remains compatible with PyTorch training
pipelines while providing a quantum‑centric contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp


class QuantumRefinement(nn.Module):
    """Quantum refinement layer implemented with EstimatorQNN."""
    def __init__(self, latent_dim: int, reps: int = 2) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # Build a parameterized circuit with RealAmplitudes
        self.circuit = RealAmplitudes(latent_dim, reps=reps)
        # Observable Y on the first qubit (extended to all qubits for compatibility)
        self.observable = SparsePauliOp.from_list([("Y" * latent_dim, 1)])
        # Estimator primitive
        estimator = Estimator()
        # EstimatorQNN uses the first parameter as input, the rest as weights
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.circuit.parameters[0]],
            weight_params=self.circuit.parameters[1:],
            estimator=estimator,
        )
        # Optimizer for quantum weights
        self.optimizer = COBYLA(maxiter=200)
        self._train_weights()

    def _train_weights(self) -> None:
        """Initial training of the quantum weights on dummy data."""
        dummy_inputs = np.zeros((1, self.latent_dim))
        loss = lambda w: np.mean((self.qnn(dummy_inputs, w) - dummy_inputs) ** 2)
        init_params = np.random.uniform(-np.pi, np.pi, len(self.circuit.parameters[1:]))
        opt_params = self.optimizer.minimize(loss, init_params)
        self.qnn.set_weights(opt_params)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for quantum evaluation
        latents_np = latents.detach().cpu().numpy()
        # Evaluate quantum circuit
        qnn_output = self.qnn(latents_np)
        # Convert back to torch
        return torch.tensor(qnn_output, dtype=torch.float32, device=latents.device)


@dataclass
class HybridAutoEncoderConfig:
    """Configuration values for :class:`HybridAutoEncoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class HybridAutoEncoder(nn.Module):
    """Hybrid autoencoder that uses a quantum refinement layer."""
    def __init__(self, config: HybridAutoEncoderConfig) -> None:
        super().__init__()
        self.encoder = self._build_encoder(config)
        self.decoder = self._build_decoder(config)
        self.quantum_refine = QuantumRefinement(config.latent_dim)

    def _build_encoder(self, config: HybridAutoEncoderConfig) -> nn.Sequential:
        layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.latent_dim))
        return nn.Sequential(*layers)

    def _build_decoder(self, config: HybridAutoEncoderConfig) -> nn.Sequential:
        layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.input_dim))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(inputs)
        z_refined = self.quantum_refine(z)
        return self.decoder(z_refined)


def train_hybrid_autoencoder(
    model: HybridAutoEncoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop that accepts a hybrid model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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


# Helper
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


__all__ = ["HybridAutoEncoder", "HybridAutoEncoderConfig", "train_hybrid_autoencoder"]
