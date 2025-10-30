"""Quantum autoencoder using swap‑test and variational circuit.

The class mimics the classical AutoencoderModel interface for hybrid workflows.
"""

import numpy as np
import torch
from torch import nn
from typing import Tuple, List, Optional

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class AutoencoderModel:
    """Quantum autoencoder with a swap‑test based latent extraction."""
    def __init__(
        self,
        latent_dim: int = 3,
        trash_dim: int = 2,
        reps: int = 5,
        seed: int = 42,
    ) -> None:
        algorithm_globals.random_seed = seed
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self.sampler = Sampler()
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=self._interpret,
            output_shape=2,
            sampler=self.sampler,
        )
        # Parameters for gradient descent
        self.optimizer = COBYLA(maxiter=200, tol=1e-4)

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a circuit with an encoder ansatz and a swap‑test."""
        n_latent = self.latent_dim
        n_trash = self.trash_dim
        qr = QuantumRegister(n_latent + 2 * n_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encoder ansatz
        qc.compose(
            RealAmplitudes(n_latent + n_trash, reps=self.reps),
            range(0, n_latent + n_trash),
            inplace=True,
        )
        qc.barrier()

        # Swap‑test to compare trash qubits
        aux = n_latent + 2 * n_trash
        qc.h(aux)
        for i in range(n_trash):
            qc.cswap(aux, n_latent + i, n_latent + n_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    @staticmethod
    def _interpret(x: np.ndarray) -> np.ndarray:
        """Identity interpret: return the raw measurement probabilities."""
        return x

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Run the quantum circuit and return the latent probabilities."""
        # For each sample, set the initial state using the trash qubits.
        # Here we simply ignore the data and return the circuit output.
        # A realistic implementation would prepare the state from data.
        probs = self.sampler.run(self.circuit).result().get_statevector()
        # Convert to real probabilities for the auxiliary qubit
        aux_prob = np.abs(probs[1]) ** 2
        return torch.tensor([aux_prob], dtype=torch.float32)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Placeholder: quantum decoder not implemented."""
        raise NotImplementedError("Quantum decoder requires a custom circuit.")

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Encode then decode; decoder is a no‑op in this toy example."""
        z = self.encode(data)
        return self.decode(z)

    # ------------------------------------------------------------------
    # Training routine using COBYLA
    # ------------------------------------------------------------------
    def train_autoencoder(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 50,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
    ) -> List[float]:
        """Train the quantum autoencoder by minimizing reconstruction loss."""
        history: List[float] = []
        for _ in range(epochs):
            # Forward pass
            preds = self.encode(data)
            # Dummy target: use the same data as target
            loss = torch.mean((preds - data) ** 2).item()
            # Update parameters via COBYLA
            def objective(params):
                # Update circuit parameters
                for name, val in zip(self.circuit.parameters, params):
                    self.circuit.set_parameter(name, val)
                # Re‑evaluate loss
                preds = self.encode(data)
                return float(torch.mean((preds - data) ** 2).item())
            params = np.array(list(self.circuit.parameters))
            new_params = self.optimizer.optimize(params, objective)
            for name, val in zip(self.circuit.parameters, new_params):
                self.circuit.set_parameter(name, val)
            history.append(loss)
        return history

__all__ = ["AutoencoderModel"]
