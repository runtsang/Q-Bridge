import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

class HybridAutoEncoder:
    """Quantum‑centric hybrid auto‑encoder that uses a RealAmplitudes ansatz,
    a domain‑wall swap‑test entanglement block, and a state‑vector sampler to
    produce a latent vector of size `latent_dim`."""
    def __init__(self, latent_dim: int = 3, trash_dim: int = 2, reps: int = 5) -> None:
        algorithm_globals.random_seed = 42
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self.sampler = Sampler()
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=(latent_dim,),
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        num_qubits = self.latent_dim + 2 * self.trash_dim + 1
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Variational block
        qc.append(RealAmplitudes(num_qubits, reps=self.reps), range(num_qubits))
        qc.barrier()

        # Domain‑wall swap‑test entanglement
        for i in range(self.trash_dim):
            qc.cswap(0, self.latent_dim + i, self.latent_dim + self.trash_dim + i)

        qc.h(num_qubits - 1)          # auxiliary qubit for measurement
        qc.measure(num_qubits - 1, cr[0])
        return qc

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Map a classical feature vector `x` to a quantum latent vector."""
        # Simple linear mapping from input to circuit parameters
        param_map = np.tanh(x @ np.random.randn(x.shape[-1], len(self.circuit.parameters)))
        self.qnn.weight_params = param_map
        return self.qnn.forward(x)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Placeholder classical decoder that maps the latent vector back to a flat image."""
        # Randomly project back to 28×28 for demonstration
        return latent @ np.random.randn(self.latent_dim, 28 * 28)

__all__ = ["HybridAutoEncoder"]
