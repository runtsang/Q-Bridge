"""Quantum hybrid autoencoder using a parameterized circuit and swap test."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class HybridAutoencoder(SamplerQNN):
    """A variational autoencoder that encodes inputs into a latent subspace
    using a RealAmplitudes ansatz and reconstructs via a swap test."""
    def __init__(
        self,
        latent_dim: int = 3,
        trash_dim: int = 2,
        reps: int = 5,
        seed: int | None = None,
    ) -> None:
        algorithm_globals.random_seed = seed or 42
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps

        # Build the parameterized circuit
        self.circuit = self._build_circuit()
        # Sampler primitive
        sampler = StatevectorSampler()
        # No input parameters; all parameters are trainable weights
        super().__init__(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=self._interpret,
            output_shape=(2,),
            sampler=sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the circuit with RealAmplitudes ansatz and swap test."""
        num_latent = self.latent_dim
        num_trash = self.trash_dim
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Ansatz on latent + first trash block
        ansatz = RealAmplitudes(num_latent + num_trash, reps=self.reps)
        circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()

        # Swap test
        aux = num_latent + 2 * num_trash
        circuit.h(aux)
        for i in range(num_trash):
            circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])
        return circuit

    def _interpret(self, result: np.ndarray) -> np.ndarray:
        """Return the raw measurement probability of the auxiliary qubit."""
        # result shape: (n_samples, 2) where columns correspond to |0>, |1>
        # We return the probability of measuring |1> (reconstruction indicator)
        return result[:, 1]

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 50,
        lr: float = 1e-3,
        optimizer_cls=COBYLA,
        seed: int | None = None,
    ) -> list[float]:
        """Train the QNN to minimise MSE between predictions and targets."""
        algorithm_globals.random_seed = seed or 42
        opt = optimizer_cls(learning_rate=lr)
        history: list[float] = []

        def loss_fn(params: np.ndarray) -> float:
            self.set_weights(params)
            preds = self.forward(data)
            loss = np.mean((preds - data) ** 2)
            return loss

        for epoch in range(epochs):
            params, loss, _ = opt.minimize(loss_fn, self.get_weights(), return_all=True)
            history.append(loss)
            if epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {loss:.6f}")
        return history

def create_hybrid_autoencoder(
    latent_dim: int = 3,
    trash_dim: int = 2,
    reps: int = 5,
    seed: int | None = None,
) -> HybridAutoencoder:
    """Convenience factory for the quantum hybrid autoencoder."""
    return HybridAutoencoder(latent_dim=latent_dim, trash_dim=trash_dim, reps=reps, seed=seed)

__all__ = [
    "HybridAutoencoder",
    "create_hybrid_autoencoder",
]
