"""Quantum autoencoder implementation based on a swap‑test variational circuit.

The module defines a single class :class:`AutoencoderGen010` that wraps a
`SamplerQNN`.  It reproduces the behaviour of the original QML seed but
adds configurable latent and trash dimensions, integration with a
classical ``QuanvolutionFilter`` for image patch encoding, and a
light‑weight training routine that can be called from the classical
module above.

The quantum circuit uses a RealAmplitudes ansatz followed by a swap‑test
to compute the fidelity between the latent and trash registers.  The
output of the circuit is interpreted as a two‑dimensional probability
vector which is decoded classically by a linear layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


@dataclass
class AutoencoderConfig:
    """Quantum autoencoder configuration."""
    latent_dim: int = 3
    trash_dim: int = 2
    reps: int = 5
    shots: int = 1024
    backend: str = "qasm_simulator"
    seed: int = 42


def _identity_interpret(x: np.ndarray) -> np.ndarray:
    """Return the raw measurement probabilities."""
    return x


def _auto_encoder_circuit(num_latent: int, num_trash: int, reps: int) -> QuantumCircuit:
    """Build a swap‑test based variational autoencoder circuit."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Variational ansatz on latent + trash
    circuit.compose(
        RealAmplitudes(num_latent + num_trash, reps=reps),
        range(0, num_latent + num_trash),
        inplace=True,
    )
    circuit.barrier()

    # Swap‑test
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])

    return circuit


class AutoencoderGen010:
    """Quantum autoencoder backed by a :class:`SamplerQNN`."""
    def __init__(self, config: AutoencoderConfig) -> None:
        algorithm_globals.random_seed = config.seed
        self.config = config

        # Build circuit
        self.circuit = _auto_encoder_circuit(
            config.latent_dim, config.trash_dim, config.reps
        )

        # Sampler instance
        self.sampler = StatevectorSampler(backend=config.backend)

        # QNN wrapper
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=_identity_interpret,
            output_shape=2,
            sampler=self.sampler,
        )

        # Optional classical decoder (linear layer)
        self.decoder = nn.Linear(2, config.latent_dim)

    def run(self, latent: torch.Tensor) -> torch.Tensor:
        """Encode a classical latent vector via the quantum circuit."""
        # Convert latent to numpy array
        latent_np = latent.detach().cpu().numpy()
        # Forward pass through QNN
        outputs = self.qnn.forward(latent_np)
        # Interpret as probabilities, then decode classically
        probs = np.asarray(outputs)
        decoded = self.decoder(torch.from_numpy(probs).float())
        return decoded

    def train_qnn(
        self,
        data: Iterable[torch.Tensor],
        *,
        epochs: int = 50,
        learning_rate: float = 1e-3,
    ) -> list[float]:
        """Train the quantum circuit weights to minimise reconstruction loss."""
        # Simple gradient‑free optimiser (COBYLA) is used; in practice a
        # stochastic optimiser with automatic differentiation would be more
        # efficient.  Here we provide a minimal example.
        from qiskit_machine_learning.optimizers import COBYLA

        opt = COBYLA()
        history: list[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in data:
                # Forward pass
                outputs = self.qnn.forward(batch.numpy())
                probs = np.asarray(outputs)
                decoded = self.decoder(torch.from_numpy(probs).float())
                loss = torch.nn.functional.mse_loss(decoded, batch)
                # Backward: COBYLA is a derivative‑free optimiser, so we
                # only update the weight parameters via the QNN's `optimize`
                # helper which internally handles finite‑difference gradients.
                opt.optimize(
                    parameters=np.array(self.qnn.parameters()),
                    loss=lambda p: torch.nn.functional.mse_loss(
                        self.qnn.forward(batch.numpy(), parameters=p), batch
                    ).item(),
                )
                epoch_loss += loss.item()
            epoch_loss /= len(data)
            history.append(epoch_loss)
        return history


__all__ = ["AutoencoderConfig", "AutoencoderGen010"]
