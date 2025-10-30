import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.utils import algorithm_globals

class QuantumQuanvolutionFilter:
    """Quantum filter that applies a random two‑qubit circuit to each 2×2 patch."""
    def __init__(self, n_wires: int = 4, n_ops: int = 8):
        self.n_wires = n_wires
        self.n_ops = n_ops
        self.encoder = RealAmplitudes(n_wires, reps=1)
        self.random_layer = RealAmplitudes(n_wires, reps=1)
        self.sampler = Sampler()

    def forward(self, patches: np.ndarray) -> np.ndarray:
        """Apply the filter to a batch of 2×2 patches.

        patches shape: (batch, 4)
        """
        qc = QuantumCircuit(self.n_wires)
        for i in range(self.n_wires):
            qc.ry(patches[:, i], i)
        qc.append(self.random_layer, range(self.n_wires))
        qc.measure_all()
        result = self.sampler.run(qc, shots=1).result()
        counts = result.get_counts()
        # Convert counts to amplitudes (placeholder)
        return np.array([int(k, 2) for k in counts.keys()])

class QuantumAutoencoder:
    """Variational quantum autoencoder using a sampler QNN."""
    def __init__(self, latent_dim: int, input_dim: int):
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        num_qubits = self.latent_dim + self.input_dim
        qc = QuantumCircuit(num_qubits)
        # Encode input into first input_dim qubits
        for i in range(self.input_dim):
            qc.h(i)
        # Variational ansatz on latent qubits
        ansatz = RealAmplitudes(self.latent_dim, reps=3)
        qc.append(ansatz, range(self.latent_dim))
        # Swap test for reconstruction
        for i in range(self.latent_dim):
            qc.cswap(i, self.input_dim + i, self.input_dim + self.latent_dim + i)
        qc.measure_all()
        return qc

    def get_qnn(self) -> SamplerQNN:
        sampler = Sampler()
        return SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=(self.latent_dim,),
            sampler=sampler,
        )

def train_quantum_autoencoder(
    qnn: SamplerQNN,
    data: torch.Tensor,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    optimizer_cls=COBYLA,
) -> list[float]:
    """Train the quantum autoencoder using a classical optimizer."""
    algorithm_globals.random_seed = 42
    opt = optimizer_cls(maxiter=epochs)
    loss_history = []

    for epoch in range(epochs):
        loss = 0.0
        for batch in data:
            output = qnn.forward(batch.numpy())
            loss += np.mean((output - batch.numpy()) ** 2)
        loss /= len(data)
        loss_history.append(loss)
        opt.minimize(lambda params: loss, qnn.parameters)
    return loss_history

__all__ = [
    "QuantumQuanvolutionFilter",
    "QuantumAutoencoder",
    "train_quantum_autoencoder",
]
