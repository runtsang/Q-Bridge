import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA

class AutoencoderHybrid:
    """Variational autoencoder built on Qiskit with a RealAmplitudes ansatz and a Qiskit sampler."""
    def __init__(
        self,
        latent_dim: int = 3,
        num_trash: int = 2,
        reps: int = 5,
    ):
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.backend = Aer.get_backend("qasm_simulator")
        self.sampler = Sampler()
        self._build_circuit()

    def _ansatz(self, num_qubits: int) -> RealAmplitudes:
        return RealAmplitudes(num_qubits, reps=self.reps)

    def _auto_encoder_circuit(self, num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)
        circ.compose(self._ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        circ.barrier()
        aux = num_latent + 2 * num_trash
        circ.h(aux)
        for i in range(num_trash):
            circ.cswap(aux, num_latent + i, num_latent + num_trash + i)
        circ.h(aux)
        circ.measure(aux, cr[0])
        return circ

    def _build_circuit(self) -> None:
        self.circuit = self._auto_encoder_circuit(self.latent_dim, self.num_trash)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode classical data into a probability of measuring |1⟩ on the auxiliary qubit."""
        circ = self.circuit.copy()
        # In a realistic setting, parameters would be conditioned on x; here we use a fixed assignment
        result = self.sampler.run(circ, shots=1024).result()
        counts = result.get_counts(circ)
        prob = sum(int(k) * v for k, v in counts.items()) / 1024
        return np.array([prob])

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode a latent vector back to a real‑valued vector (placeholder transformation)."""
        return 2 * z - 1

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        z = self.encode(x)
        return self.decode(z)

    def train(
        self,
        data: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-2,
    ) -> None:
        """Classical optimizer over the variational parameters using COBYLA."""
        params = np.random.randn(self.circuit.num_parameters)
        opt = COBYLA(maxfun=epochs)

        def objective(p: np.ndarray) -> float:
            self.circuit.set_parameters(p)
            recon = self.reconstruct(data)
            loss = np.mean((recon - data) ** 2)
            return loss

        opt.optimize(len(params), objective, initial_point=params)
        self.circuit.set_parameters(opt.x)

__all__ = ["AutoencoderHybrid"]
