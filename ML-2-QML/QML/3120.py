import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA

class AutoencoderGen044:
    """Qiskit variational autoencoder with swap‑test kernel evaluation."""
    def __init__(self, latent_dim: int = 3, num_trash: int = 2, reps: int = 5, seed: int = 42) -> None:
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.sim = Aer.get_backend('aer_simulator')
        self.ansatz = RealAmplitudes(latent_dim + num_trash, reps=reps)
        self._build_swap_test()

    def _build_swap_test(self) -> None:
        """Swap‑test circuit for measuring overlap."""
        self.swap_test = QuantumCircuit(self.latent_dim + self.num_trash + 1, 1)
        aux = self.latent_dim + self.num_trash
        self.swap_test.h(aux)
        for i in range(self.num_trash):
            self.swap_test.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        self.swap_test.h(aux)
        self.swap_test.measure(aux, 0)

    def encode(self, x: np.ndarray) -> Statevector:
        """Encode a classical vector into a quantum state."""
        qc = self.ansatz.bind_parameters(x)
        return Statevector.from_instruction(qc)

    def decode(self, sv: Statevector) -> np.ndarray:
        """Return a classical reconstruction from a statevector."""
        probs = np.abs(sv.data)**2
        return probs

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full encode‑decode pass."""
        sv = self.encode(x)
        return self.decode(sv)

    def kernel_matrix(self, xs: np.ndarray, refs: np.ndarray) -> np.ndarray:
        """Compute Gram matrix via fidelity between encoded inputs and reference vectors."""
        gram = np.zeros((len(xs), len(refs)))
        for i, x in enumerate(xs):
            sv_x = self.encode(x)
            for j, r in enumerate(refs):
                sv_r = self.encode(r)
                fidelity = np.abs(np.vdot(sv_x.data, sv_r.data))**2
                gram[i, j] = fidelity
        return gram

    def train(self, data: np.ndarray, epochs: int = 10, lr: float = 0.01) -> None:
        """Variational training using COBYLA on reconstruction loss."""
        opt = COBYLA(maxiter=epochs)
        def loss_fn(params):
            total = 0.0
            for x in data:
                sv = self.encode(x)
                recon = self.decode(sv)
                total += np.linalg.norm(recon - x)**2
            return total / len(data)
        opt.optimize(num_vars=len(self.ansatz.parameters), objective_function=loss_fn, initial_point=np.array(self.ansatz.parameters))
        self.ansatz.parameters = opt.x

__all__ = ["AutoencoderGen044"]
