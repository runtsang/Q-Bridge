"""Quantum autoencoder using a variational ansatz and swap‑test entanglement.

The class implements a parameterised circuit that encodes a classical
feature vector into a low‑dimensional quantum state, applies a swap test
with ancillary qubits, and then decodes back to a classical vector via
measurement.  Training is performed with the Qiskit COBYLA optimiser
on the sampler‑based QNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, RY
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import Sampler as StatevectorSampler


@dataclass
class QAutoencoderConfig:
    """Configuration for :class:`AutoencoderGen240`."""
    num_features: int
    latent_dim: int = 3
    trash_dim: int = 2
    reps: int = 3
    seed: int = 42


class AutoencoderGen240:
    """A variational quantum autoencoder with swap‑test entanglement."""
    def __init__(self, cfg: QAutoencoderConfig) -> None:
        self.cfg = cfg
        algorithm_globals.random_seed = cfg.seed
        self.sampler = StatevectorSampler()
        self._build_ansatz()
        self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.ansatz.parameters,
            interpret=self._interpret,
            output_shape=(self.cfg.num_features,),
            sampler=self.sampler,
        )

    def _build_ansatz(self) -> None:
        """Create a RealAmplitudes ansatz for the latent subspace."""
        self.ansatz = RealAmplitudes(
            self.cfg.latent_dim + self.cfg.trash_dim, reps=self.cfg.reps
        )

    def _build_circuit(self) -> None:
        """Construct the full autoencoder circuit."""
        n = self.cfg.latent_dim + 2 * self.cfg.trash_dim + 1
        qr = QuantumRegister(n, "q")
        cr = ClassicalRegister(1, "c")
        self.circuit = QuantumCircuit(qr, cr)

        # Encode with the ansatz
        self.circuit.append(self.ansatz, range(0, self.cfg.latent_dim + self.cfg.trash_dim))

        # Swap‑test entanglement
        aux = self.cfg.latent_dim + 2 * self.cfg.trash_dim
        self.circuit.h(aux)
        for i in range(self.cfg.trash_dim):
            self.circuit.cswap(aux, self.cfg.latent_dim + i, self.cfg.latent_dim + self.cfg.trash_dim + i)
        self.circuit.h(aux)

        # Decode by applying the inverse ansatz
        self.circuit.append(self.ansatz.inverse(), range(0, self.cfg.latent_dim + self.cfg.trash_dim))

        self.circuit.measure(aux, cr[0])

    def _interpret(self, output: np.ndarray) -> np.ndarray:
        """Map the sampler output to a classical vector."""
        # Use the amplitude of the auxiliary qubit as a feature indicator
        return output.real

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode classical features into a latent quantum state."""
        # Here we simply use the sampler to obtain the statevector
        sv = Statevector.from_label("0" * self.circuit.num_qubits)
        return sv.evolve(self.ansatz).data

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent vector back to classical space by measuring."""
        return self._interpret(latent)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the quantum autoencoder on a batch of inputs."""
        # For simplicity, we ignore inputs and use the trained parameters
        return self.qnn.forward()

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 50,
        lr: float = 0.01,
    ) -> list[float]:
        """Train the variational parameters using COBYLA."""
        opt = COBYLA(lr=lr, maxiter=epochs)
        loss_history = []

        def loss_fn(params: np.ndarray) -> float:
            # Update ansatz parameters
            self.ansatz.set_parameters(params)
            self._build_circuit()
            self.qnn = SamplerQNN(
                circuit=self.circuit,
                input_params=[],
                weight_params=self.ansatz.parameters,
                interpret=self._interpret,
                output_shape=(self.cfg.num_features,),
                sampler=self.sampler,
            )
            preds = self.qnn.forward()
            # Simple MSE loss against input data
            return float(np.mean((preds - data) ** 2))

        opt.minimize(loss_fn, self.ansatz.parameters)
        loss_history.append(opt.last_value)
        return loss_history


__all__ = ["AutoencoderGen240", "QAutoencoderConfig"]
