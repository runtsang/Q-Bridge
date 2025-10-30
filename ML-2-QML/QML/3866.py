from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QuantumSamplerQNN
from qiskit_machine_learning.optimizers import COBYLA, Adam
from qiskit_machine_learning.utils import algorithm_globals

class HybridAutoencoder:
    """Quantum-centric autoencoder that maps classical data into a quantum
    circuit, applies a variational ansatz, and decodes via a quantum sampler."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        reps: int = 3,
        optimizer: str = "COBYLA",
    ) -> None:
        algorithm_globals.random_seed = 42
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reps = reps
        self.optimizer_name = optimizer

        self.feature_params = ParameterVector("x", input_dim)
        self.circuit = QuantumCircuit(input_dim)
        self.circuit.append(RealAmplitudes(input_dim, reps=reps), range(input_dim))

        self.ansatz = RealAmplitudes(latent_dim, reps=reps)
        self.circuit.append(self.ansatz, range(latent_dim))

        self.sampler = StatevectorSampler()
        self.qnn = QuantumSamplerQNN(
            circuit=self.circuit,
            input_params=self.feature_params,
            weight_params=self.ansatz.parameters,
            interpret=lambda x: x,
            output_shape=self.input_dim,
            sampler=self.sampler,
        )

        if optimizer == "COBYLA":
            self.optimizer = COBYLA()
        else:
            self.optimizer = Adam()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Return reconstructed inputs for a batch."""
        batch_size = inputs.shape[0]
        reconstructs = []
        for i in range(batch_size):
            param_dict = dict(zip(self.feature_params, inputs[i]))
            output = self.qnn(param_dict)
            reconstructs.append(np.array(output).reshape(self.input_dim))
        return np.stack(reconstructs)

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> list[float]:
        """Train the variational parameters using a classical optimizer."""
        history: list[float] = []
        for _ in range(epochs):
            loss = 0.0
            for x in data:
                param_dict = dict(zip(self.feature_params, x))
                recon = self.qnn(param_dict)
                loss += np.mean((np.array(recon) - x) ** 2)
            loss /= len(data)
            history.append(loss)
            # Simple optimizer step; in practice, gradients should be computed
            self.optimizer.step(self.qnn, loss, lr)
        return history

__all__ = ["HybridAutoencoder"]
