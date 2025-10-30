"""Pennylane implementation of a quantum autoencoder.

The design mirrors the classical counterpart:
* `Autoencoder` builds a variational circuit with an encoder, a
  swap‑test based trash‑qubit discarding, and a decoder.
* `train` optimises the circuit parameters to minimise a fidelity‑based
  reconstruction loss.
* `encode` and `decode` expose the latent representation and the
  reconstructed state, respectively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pennylane as qml
import numpy as np
import torch
from pennylane import numpy as pnp
from pennylane import qnode
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
from pennylane.measurements import StateFn


@dataclass
class AutoencoderConfig:
    """Configuration for the quantum autoencoder."""
    num_qubits: int
    latent_dim: int
    trash_dim: int
    reps: int = 2
    device: str = "default.qubit"
    shots: int = 1024
    opt_type: str = "adam"
    lr: float = 0.01
    epochs: int = 200
    seed: int = 42


class Autoencoder:
    """Quantum autoencoder built with Pennylane."""
    def __init__(self, config: AutoencoderConfig) -> None:
        self.config = config
        qml.set_options(device=config.device, shots=config.shots, seed=config.seed)
        self.params = pnp.random.uniform(0, 2 * np.pi, config.reps * config.num_qubits * 3)
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Construct the variational autoencoder circuit."""
        @qml.qnode(qml.device(self.config.device, wires=self.config.num_qubits), interface="torch")
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encode the classical data via amplitude encoding
            qml.AmplitudeEmbedding(
                features=inputs,
                wires=range(self.config.latent_dim),
                normalize=True,
            )
            # Trash qubits start after the latent block
            trash_wires = range(self.config.latent_dim, self.config.latent_dim + self.config.trash_dim)
            # Encoder block
            for r in range(self.config.reps):
                for w in range(self.config.num_qubits):
                    qml.RX(params[r * self.config.num_qubits + w], wires=w)
                for w in range(self.config.num_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
            # Swap‑test to discard trash qubits
            aux_wire = self.config.num_qubits
            qml.H(aux_wire)
            for i, t in enumerate(trash_wires):
                qml.CSWAP([aux_wire, self.config.latent_dim + i, t])
            qml.H(aux_wire)
            # Decoder block (reverse of encoder)
            for r in reversed(range(self.config.reps)):
                for w in range(self.config.num_qubits):
                    qml.RX(params[r * self.config.num_qubits + w], wires=w)
                for w in range(self.config.num_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
            # Measurement of the latent qubits
            return qml.expval(StateFn(qml.PauliZ(wires=self.config.latent_dim)))
        self.circuit = circuit

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Return the latent representation of the input data."""
        return self.circuit(data, self.params)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input from the latent representation."""
        # In this toy example we simply return the latent back
        return latent

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encoding, swapping, decoding."""
        latent = self.encode(data)
        return self.decode(latent)

    def _loss(self, data: torch.Tensor) -> torch.Tensor:
        """Mean fidelity loss between input and reconstruction."""
        recon = self.forward(data)
        # Fidelity for pure states: |<psi|phi>|^2
        fidelity = torch.mean((data * recon).sum(dim=1) ** 2)
        return 1 - fidelity

    def train(self, data: torch.Tensor) -> None:
        """Optimise the circuit parameters to minimise the reconstruction loss."""
        if self.config.opt_type == "adam":
            opt = AdamOptimizer(self.config.lr)
        else:
            opt = GradientDescentOptimizer(self.config.lr)

        for epoch in range(self.config.epochs):
            loss_val = self._loss(data)
            opt.step(lambda _: loss_val, self.params)
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: loss={loss_val.item():.6f}")

    def evaluate(self, data: torch.Tensor) -> float:
        """Return average reconstruction MSE."""
        with torch.no_grad():
            recon = self.forward(data)
            mse = torch.mean((data - recon) ** 2).item()
        return mse


def Autoencoder(
    *,
    num_qubits: int,
    latent_dim: int,
    trash_dim: int,
    reps: int = 2,
    device: str = "default.qubit",
    shots: int = 1024,
    opt_type: str = "adam",
    lr: float = 0.01,
    epochs: int = 200,
    seed: int = 42,
) -> Autoencoder:
    """Convenience factory mirroring the classical interface."""
    cfg = AutoencoderConfig(
        num_qubits=num_qubits,
        latent_dim=latent_dim,
        trash_dim=trash_dim,
        reps=reps,
        device=device,
        shots=shots,
        opt_type=opt_type,
        lr=lr,
        epochs=epochs,
        seed=seed,
    )
    return Autoencoder(cfg)


__all__ = ["Autoencoder", "AutoencoderConfig", "Autoencoder"]
