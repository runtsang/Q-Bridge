"""Hybrid classical‑quantum regressor.

The module exposes a single :class:`HybridEstimatorQNN` that
* encodes the raw input with a small autoencoder,
* feeds the latent representation into a variational quantum circuit,
* returns the expectation value of Z on the first qubit as the regression output.

The quantum circuit is defined in :mod:`HybridEstimatorQNN_qml` and is instantiated on
the fly.  The quantum parameters are wrapped in :class:`torch.nn.Parameter` so that
gradient descent can be applied through the :class:`torch.autograd` engine.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import grad

from.Autoencoder import Autoencoder, AutoencoderConfig
from.HybridEstimatorQNN_qml import hybrid_quantum_circuit
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

# --------------------------------------------------------------------------- #
#   Classical encoder
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    """Thin wrapper around the auto‑encoder that exposes only the encoder part."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
        # Keep only the encoder
        self.encoder = self.autoencoder.encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

# --------------------------------------------------------------------------- #
#   Hybrid estimator
# --------------------------------------------------------------------------- #
class HybridEstimatorQNN(nn.Module):
    """
    A two‑stage model:

    1. Classical encoder (auto‑encoder) reduces the dimensionality.
    2. Quantum variational circuit consumes the latent vector and returns a single
       expectation value that is interpreted as the regression output.
    """
    def __init__(
        self,
        input_dim: int = 2,
        latent_dim: int = 4,
        hidden_dims: tuple[int, int] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Classical encoder
        self.encoder = Encoder(
            AutoencoderConfig(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        )

        # Quantum part
        self.quantum_circuit = hybrid_quantum_circuit(latent_dim)
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.quantum_circuit,
            observables=self.quantum_circuit.num_qubits * [("Z", 1)],
            input_params=self.quantum_circuit.parameters[:latent_dim],
            weight_params=self.quantum_circuit.parameters[latent_dim:],
            estimator=self.estimator,
        )

        # Wrap quantum weights as trainable parameters
        self.q_weights = nn.Parameter(
            torch.tensor([p.value() for p in self.quantum_circuit.parameters[latent_dim:]],
                         dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar output for each batch element."""
        # Encode classically
        latent = self.encoder(x)

        # Bind quantum parameters
        weight_bindings = dict(zip(self.quantum_circuit.parameters[latent.shape[1]:], self.q_weights))
        # Evaluate quantum circuit
        q_out = self.qnn(latent, weight_bindings)
        # Output is a batch of expectation values
        return q_out.squeeze(-1)

    def parameters(self, recurse: bool = True):
        """Yield all trainable parameters (encoder + quantum)."""
        for p in super().parameters(recurse):
            yield p
        yield self.q_weights

__all__ = ["HybridEstimatorQNN"]
