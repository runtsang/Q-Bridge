import numpy as np
from typing import List
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StateSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.utils import algorithm_globals

class AutoencoderQNN:
    """Hybrid variational autoencoder implemented with Qiskit."""
    def __init__(self,
                 num_latent: int = 3,
                 num_trash: int = 2,
                 reps: int = 5,
                 seed: int = 42):
        algorithm_globals.random_seed = seed
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.sampler = StateSampler()
        self.circuit = self._build_circuit()

        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        total_qubits = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode data using a parameter‑shared RealAmplitudes ansatz
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        qc.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)

        # Swap‑test style entanglement with trash qubits
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        """Forward pass through the sampler QNN."""
        return self.qnn.forward(params)

    def train(self,
              initial_params: np.ndarray,
              data: np.ndarray,
              epochs: int = 20,
              learning_rate: float = 0.1,
              weight_decay: float = 0.0) -> List[float]:
        """Gradient‑free training loop using COBYLA."""
        opt = COBYLA()
        history: List[float] = []

        def loss_fn(params):
            preds = self.evaluate(params)
            loss = np.mean((preds - data) ** 2)
            return loss

        params = initial_params
        for _ in range(epochs):
            params = opt.minimize(loss_fn, params)
            loss = loss_fn(params)
            history.append(loss)
        return history

__all__ = ["AutoencoderQNN"]
