"""Quantum autoencoder implementation using Qiskit.
Combines a RealAmplitudes ansatz, swap‑test based latent extraction and a
SamplerQNN style measurement for the latent vector."""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class HybridAutoencoderQuantum:
    """Quantum autoencoder that accepts an input statevector, encodes it
    into a low‑dimensional latent space, and reconstructs the original
    state via a swap‑test style circuit."""
    def __init__(self,
                 num_latent: int = 3,
                 num_trash: int = 2,
                 reps: int = 5,
                 seed: int | None = 42) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        algorithm_globals.random_seed = seed
        self.sampler = Sampler()
        self.circuit = self._build_circuit()
        self.qnn = self._build_qnn()

    def _ansatz(self, num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=self.reps)

    def _auto_encoder_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.compose(self._ansatz(self.num_latent + self.num_trash), range(0, self.num_latent + self.num_trash), inplace=True)
        qc.barrier()
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        return self._auto_encoder_circuit()

    def _build_qnn(self) -> SamplerQNN:
        """Wrap the auto‑encoder circuit as a SamplerQNN."""
        return SamplerQNN(circuit=self.circuit,
                          input_params=[],
                          weight_params=self.circuit.parameters,
                          sampler=self.sampler)

    def encode(self, state: Statevector) -> np.ndarray:
        """Encode a classical statevector into a probability distribution over latent states."""
        job = self.sampler.run(self.circuit, shots=1024, input_state=state)
        counts = job.result().get_counts()
        probs = np.zeros(self.num_latent + 1)
        for bit, cnt in counts.items():
            probs[int(bit)] = cnt / 1024
        return probs

    def decode(self, latent_probs: np.ndarray) -> Statevector:
        """Reconstruct the state from latent probabilities via a simple inverse circuit."""
        init = Statevector.from_label('0' * self.circuit.num_qubits)
        return init.evolve(self.circuit)

    def train(self,
              data: list[Statevector],
              *,
              epochs: int = 20,
              learning_rate: float = 0.1) -> list[float]:
        """Gradient‑free training using COBYLA to minimise reconstruction error."""
        opt = COBYLA(maxiter=epochs * 10, tol=1e-3)
        history: list[float] = []

        def objective(params):
            for p, val in zip(self.circuit.parameters, params):
                p.assign(val)
            loss = 0.0
            for sv in data:
                recon = self.decode(self.encode(sv))
                loss += np.linalg.norm(sv.data - recon.data)
            return loss / len(data)

        init_params = np.random.rand(len(self.circuit.parameters))
        result = opt.minimize(objective, init_params)
        history.append(result.fun)
        return history

__all__ = ["HybridAutoencoderQuantum"]
