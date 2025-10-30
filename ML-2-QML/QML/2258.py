import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler, StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

__all__ = ["QuantumAutoencoderCircuit", "get_quantum_autoencoder",
           "QuantumKernel", "quantum_kernel_matrix"]

class QuantumAutoencoderCircuit:
    """Builds a variational autoencoder circuit with swap‑test based reconstruction."""
    def __init__(self, latent_dim: int = 3, trash_dim: int = 2, reps: int = 5):
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps

    def _ansatz(self, num_qubits: int) -> QuantumCircuit:
        """Variational ansatz using RealAmplitudes."""
        return RealAmplitudes(num_qubits, reps=self.reps)

    def circuit(self) -> QuantumCircuit:
        """Return a full autoencoder circuit."""
        num_qubits = self.latent_dim + 2 * self.trash_dim + 1
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Encode data into first latent+trash qubits
        circuit.compose(self._ansatz(self.latent_dim + self.trash_dim),
                        range(0, self.latent_dim + self.trash_dim),
                        inplace=True)

        circuit.barrier()

        # Auxiliary qubit for swap test
        aux = self.latent_dim + 2 * self.trash_dim
        circuit.h(aux)
        for i in range(self.trash_dim):
            circuit.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])
        return circuit

def get_quantum_autoencoder(latent_dim: int = 3, trash_dim: int = 2,
                            reps: int = 5) -> SamplerQNN:
    """Return a SamplerQNN that implements the autoencoder circuit."""
    qa = QuantumAutoencoderCircuit(latent_dim, trash_dim, reps)
    qc = qa.circuit()
    sampler = Sampler()
    # No input parameters; all weights are variational
    qnn = SamplerQNN(circuit=qc,
                     input_params=[],
                     weight_params=qc.parameters,
                     interpret=lambda x: x,
                     output_shape=2,
                     sampler=sampler)
    return qnn

class QuantumKernel:
    """Quantum kernel using a simple Ry‑encoding ansatz."""
    def __init__(self, n_wires: int = 4):
        self.n_wires = n_wires
        self.sampler = StatevectorSampler()

    def _encode(self, x: np.ndarray) -> QuantumCircuit:
        """Encode a vector x into a circuit using Ry rotations."""
        qc = QuantumCircuit(self.n_wires)
        for i, val in enumerate(x):
            qc.ry(val, i)
        return qc

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the kernel value k(x, y) = |⟨ψ(x)|ψ(y)⟩|²."""
        sv_x = self.sampler.run(self._encode(x)).result().get_statevector()
        sv_y = self.sampler.run(self._encode(y)).result().get_statevector()
        return np.abs(np.vdot(sv_x, sv_y)) ** 2

def quantum_kernel_matrix(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
    """Compute Gram matrix using QuantumKernel."""
    kernel = QuantumKernel()
    return np.array([[kernel.kernel(x, y) for y in b] for x in a])
