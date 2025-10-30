"""Quantum autoencoder with variational ansatz and state‑vector sampler.

This module implements a hybrid quantum auto‑encoder based on a
RealAmplitudes variational circuit.  The network uses the
`SamplerQNN` from Qiskit Machine Learning and is trained with
the COBYLA optimizer on a state‑vector simulator.  The API
mirrors the classical counterpart: :func:`Autoencoder` creates an
instance, and :func:`train_autoencoder` returns the loss history.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional, Sequence

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, RawFeatureVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderHybrid`."""
    input_dim: int
    latent_dim: int = 3
    num_qubits: Optional[int] = None  # defaults to input_dim
    entanglement: str = "full"        # or "linear"
    reps: int = 3
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 50

# --------------------------------------------------------------------------- #
# Autoencoder
# --------------------------------------------------------------------------- #
class AutoencoderHybrid:
    """Quantum auto‑encoder based on a variational circuit."""
    def __init__(self, config: AutoencoderConfig) -> None:
        self.config = config
        self.num_qubits = config.num_qubits or config.input_dim
        self.feature_map = RawFeatureVector(
            num_qubits=self.num_qubits,
            input_dim=config.input_dim,
        )
        self.ansatz = RealAmplitudes(
            num_qubits=self.num_qubits,
            reps=config.reps,
            entanglement=config.entanglement,
        )
        self._build_circuit()
        algorithm_globals.random_seed = 42
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=list(self.feature_map.parameters),
            weight_params=self.ansatz.parameters,
            interpret=lambda x: x,
            output_shape=(config.latent_dim,),
            sampler=self.sampler,
        )
        self.optimizer = COBYLA()

    # ----------------------------------------------------------------------- #
    def _build_circuit(self) -> None:
        self.circuit = QuantumCircuit(self.num_qubits)
        # Feature map
        self.circuit.compose(self.feature_map, inplace=True)
        # Variational ansatz
        self.circuit.compose(self.ansatz, inplace=True)
        # Measurement of the first `latent_dim` qubits
        for i in range(self.config.latent_dim):
            self.circuit.measure(i, i)

    # ----------------------------------------------------------------------- #
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode classical data `x` into the quantum latent representation."""
        return self.qnn.forward(x)

    # ----------------------------------------------------------------------- #
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Placeholder for a classical decoder – here we simply return the latent vector."""
        return z

    # ----------------------------------------------------------------------- #
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass: encode + decode."""
        latent = self.encode(x)
        return self.decode(latent)

    # ----------------------------------------------------------------------- #
    def train_autoencoder(
        self,
        data: np.ndarray,
        *,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
    ) -> List[float]:
        """Train the quantum auto‑encoder using COBYLA."""
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        lr = lr if lr is not None else self.config.lr

        # Flatten data into batches
        dataset = np.array_split(data, max(1, len(data) // batch_size))

        history: List[float] = []

        def loss_fn(weights: np.ndarray) -> float:
            self.qnn.set_parameters(weights)
            preds = self.qnn.forward(data)
            return np.mean((preds - data) ** 2)

        # COBYLA optimisation
        best_weights, best_loss = self.optimizer.optimize(
            maxiter=epochs,
            initial_point=np.array(list(self.qnn.parameters)),
            objective_function=loss_fn,
        )
        self.qnn.set_parameters(best_weights)
        history.append(best_loss)
        return history

# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 3,
    num_qubits: Optional[int] = None,
    entanglement: str = "full",
    reps: int = 3,
) -> AutoencoderHybrid:
    """Return a configured :class:`AutoencoderHybrid`."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_qubits=num_qubits,
        entanglement=entanglement,
        reps=reps,
    )
    return AutoencoderHybrid(config)

__all__ = [
    "Autoencoder",
    "AutoencoderHybrid",
    "AutoencoderConfig",
]
