import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler

class QuantumQLSTM:
    """Quantum LSTM cell where each gate is a small variational circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget_ansatz = RealAmplitudes(n_qubits, reps=2)
        self.input_ansatz = RealAmplitudes(n_qubits, reps=2)
        self.update_ansatz = RealAmplitudes(n_qubits, reps=2)
        self.output_ansatz = RealAmplitudes(n_qubits, reps=2)

    def _apply_gate(self, ansatz: RealAmplitudes, params: np.ndarray) -> np.ndarray:
        qc = QuantumCircuit(self.n_qubits)
        qc.append(ansatz, range(self.n_qubits))
        qc.measure_all()
        backend = Aer.get_backend("qasm_simulator")
        job = qiskit.execute(qc, backend=backend, shots=1024)
        result = job.result()
        counts = result.get_counts()
        exp = 0
        for bitstring, cnt in counts.items():
            exp += ((-1) ** int(bitstring[-1])) * cnt
        return exp / sum(counts.values())

    def forward(self, inputs: np.ndarray, states: tuple[np.ndarray, np.ndarray] | None = None):
        hx, cx = states if states is not None else (np.zeros(self.hidden_dim), np.zeros(self.hidden_dim))
        # Placeholder: return classical hidden state unchanged
        return hx, (hx, cx)

class QuantumAutoencoder:
    """Variational quantum auto‑encoder using a swap‑test."""
    def __init__(self, latent_dim: int, trash_dim: int):
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.num_qubits = latent_dim + 2 * trash_dim + 1

    def circuit(self, data_vector: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.initialize(data_vector, qr[:self.latent_dim])
        ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=3)
        qc.append(ansatz, list(range(self.latent_dim + self.trash_dim)))
        qc.h(qr[self.latent_dim + self.trash_dim])
        for i in range(self.trash_dim):
            qc.cswap(qr[self.latent_dim + self.trash_dim], qr[i], qr[self.latent_dim + i])
        qc.h(qr[self.latent_dim + self.trash_dim])
        qc.measure(qr[self.latent_dim + self.trash_dim], cr[0])
        return qc

    def qnn(self, data_vector: np.ndarray) -> SamplerQNN:
        qc = self.circuit(data_vector)
        sampler = StatevectorSampler()
        qnn = SamplerQNN(
            circuit=qc,
            input_params=[],
            weight_params=qc.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=sampler,
        )
        return qnn

__all__ = ["QuantumQLSTM", "QuantumAutoencoder"]
