import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class HybridAutoencoder:
    """Quantum autoencoder that uses a swap‑test circuit and a quantum kernel."""
    def __init__(self,
                 latent_dim: int = 3,
                 trash_dim: int = 2,
                 reps: int = 5,
                 backend: str = "qasm_simulator",
                 shots: int = 1024):
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self.backend = Aer.get_backend(backend)
        self.shots = shots
        self.sampler = Sampler()
        self._build_circuit()
        self.qnn = SamplerQNN(circuit=self.circuit,
                              input_params=[],
                              weight_params=list(self.circuit.parameters),
                              interpret=self._interpret,
                              output_shape=1,
                              sampler=self.sampler)

    def _build_circuit(self):
        num_qubits = self.latent_dim + 2 * self.trash_dim + 1
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        self.circuit = QuantumCircuit(qr, cr)

        # Encode latent part with RealAmplitudes ansatz
        ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=self.reps)
        self.circuit.compose(ansatz, range(0, self.latent_dim + self.trash_dim), inplace=True)

        # Optional domain wall: flip the first half of trash qubits
        for i in range(self.latent_dim, self.latent_dim + self.trash_dim):
            self.circuit.x(i)

        # Swap‑test with trash qubits
        aux = self.latent_dim + 2 * self.trash_dim
        self.circuit.h(aux)
        for i in range(self.trash_dim):
            self.circuit.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        self.circuit.h(aux)
        self.circuit.measure(aux, cr[0])

    def _interpret(self, x):
        """Return the probability of measuring the ancilla in state |0>."""
        return x[0]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map classical data to circuit parameters."""
        n_params = len(self.circuit.parameters)
        return x.reshape(-1, n_params)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Identity decoder for placeholder."""
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the quantum autoencoder and return the fidelity estimate."""
        params = self.encode(x)
        out = self.qnn(params)
        return out

def train_hybrid_autoencoder(model: HybridAutoencoder,
                             data: torch.Tensor,
                             *,
                             epochs: int = 50,
                             lr: float = 0.01,
                             device: str = "cpu") -> list[float]:
    """Train the quantum autoencoder with COBYLA minimising reconstruction loss."""
    algorithm_globals.random_seed = 42
    opt = COBYLA(maxiter=1000)
    history = []

    def loss_fn(params):
        out = model.forward(torch.tensor(params, dtype=torch.float32))
        loss = ((out - 1.0) ** 2).mean().item()
        return loss

    init = np.random.uniform(-np.pi, np.pi, size=len(list(model.circuit.parameters)))
    for _ in range(epochs):
        res = opt.minimize(loss_fn, init)
        init = res.get('x', res.x)
        loss = loss_fn(init)
        history.append(loss)
    return history

__all__ = ["HybridAutoencoder", "train_hybrid_autoencoder"]
