import numpy as np
from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA, SPSA

class Autoencoder:
    """Quantum autoencoder based on a variational RealAmplitudes ansatz."""

    def __init__(self, input_dim: int, latent_dim: int = 3, reps: int = 2):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reps = reps
        self.sampler = Sampler()
        self._build_circuit()
        self._create_qnn()

    def _build_circuit(self):
        # Feature map: RY(θ_i) on each qubit
        self.input_params = [f"θ_{i}" for i in range(self.input_dim)]
        self.encoding_circuit = QuantumCircuit(self.input_dim)
        for i in range(self.input_dim):
            self.encoding_circuit.ry(self.input_params[i], i)

        # Variational circuit
        self.ansatz = RealAmplitudes(self.input_dim, reps=self.reps, entanglement="full")
        self.weight_params = list(self.ansatz.parameters)

        # Full circuit
        self.circuit = QuantumCircuit(self.input_dim)
        self.circuit.compose(self.encoding_circuit, range(self.input_dim), inplace=True)
        self.circuit.compose(self.ansatz, range(self.input_dim), inplace=True)

    def _create_qnn(self):
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            interpret=lambda x: x,
            output_shape=(self.input_dim,),
            sampler=self.sampler,
        )

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Return the latent representation (first m expectation values)."""
        if x.shape[0]!= self.input_dim:
            raise ValueError(f"Input vector must have length {self.input_dim}")
        outputs = self.qnn.predict([x])[0]
        return outputs[: self.latent_dim]

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct the full input from a latent vector."""
        if z.shape[0]!= self.latent_dim:
            raise ValueError(f"Latent vector must have length {self.latent_dim}")
        full_input = np.zeros(self.input_dim)
        full_input[: self.latent_dim] = z
        outputs = self.qnn.predict([full_input])[0]
        return outputs

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full autoencoder forward pass."""
        return self.decode(self.encode(x))

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Scale data to the range [-1, 1] for RY encoding."""
        data_norm = (data - data.mean(axis=0)) / data.std(axis=0)
        return np.clip(data_norm, -1.0, 1.0)

    def train(self,
              data: np.ndarray,
              epochs: int = 100,
              lr: float = 0.01,
              optimizer_name: str = "COBYLA") -> None:
        """Train the variational parameters to minimise reconstruction MSE."""
        data_norm = self._normalize(data)

        def loss_fn(weights: List[float]) -> float:
            self.qnn.set_weights(weights)
            outputs = self.qnn.predict(data_norm)
            return float(np.mean((outputs - data_norm) ** 2))

        # Choose optimizer
        if optimizer_name.upper() == "COBYLA":
            opt = COBYLA()
        else:
            opt = SPSA()

        # Run optimisation
        opt.minimize(
            loss_fn,
            self.qnn.weight_params,
            epochs=epochs,
            lr=lr,
        )

    def evaluate(self, data: np.ndarray) -> float:
        """Return the mean‑squared‑error over the dataset."""
        data_norm = self._normalize(data)
        outputs = self.qnn.predict(data_norm)
        return float(np.mean((outputs - data_norm) ** 2))

__all__ = ["Autoencoder"]
