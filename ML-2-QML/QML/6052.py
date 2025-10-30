"""Hybrid quantum autoencoder using Qiskit and Pennylane."""

import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, RawFeatureVector
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


class HybridAutoencoder:
    """Quantum autoencoder with domain‑wall injection and swap‑test reconstruction.

    The class builds a variational circuit that can be trained with a COBYLA
    optimizer.  It exposes the same encode/decode/forward API as the
    classical counterpart, making it easy to swap implementations.
    """

    def __init__(
        self,
        num_latent: int,
        num_trash: int,
        reps: int = 5,
        seed: int = 42,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.seed = seed
        algorithm_globals.random_seed = seed
        self.sampler = Sampler()
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Construct the variational autoencoder circuit."""
        # Registers
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Feature map on latent qubits
        self.feature_map = RawFeatureVector(self.num_latent)
        qc.compose(self.feature_map, range(0, self.num_latent), inplace=True)

        # Variational ansatz on latent + trash qubits
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        qc.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)

        qc.barrier()

        # Domain‑wall injection on the extra trash qubits
        for i in range(self.num_trash):
            qc.x(self.num_latent + i)

        # Swap‑test with auxiliary qubit
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        self.circuit = qc
        self.weight_params = qc.parameters
        self.input_params = self.feature_map.parameters

        # QNN wrapper
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def encode(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Encode a classical vector using the feature map."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return self.qnn.forward(x)

    def decode(self, z: np.ndarray | torch.Tensor) -> np.ndarray:
        """Decode by running the full circuit and sampling."""
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        return self.qnn.forward(z)

    def forward(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Full autoencoder: encode then decode."""
        return self.decode(self.encode(x))

    def train(self, data: np.ndarray, *, epochs: int = 50, lr: float = 0.1) -> list[float]:
        """Train the variational parameters using COBYLA to minimise MSE."""
        history: list[float] = []
        opt = COBYLA()
        weights = np.random.rand(len(self.weight_params)) * 2 * np.pi

        def cost_function(params: np.ndarray) -> float:
            self.qnn.set_weights(params)
            preds = self.forward(data)
            return np.mean((preds - data) ** 2)

        for _ in range(epochs):
            weights = opt.minimize(cost_function, weights)
            loss = cost_function(weights)
            history.append(loss)
        return history

    def evaluate(self, data: np.ndarray) -> float:
        """Return the mean‑squared reconstruction error."""
        preds = self.forward(data)
        return np.mean((preds - data) ** 2)


__all__ = ["HybridAutoencoder"]
