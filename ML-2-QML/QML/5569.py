import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector, SparsePauliOp

class AutoencoderGen475:
    """
    Quantum autoencoder that consumes a classical latent vector produced by the
    transformer‑based AutoencoderGen475 and compresses it into a smaller set of
    qubits using a swap‑test based circuit.  The circuit can also be inverted
    to reconstruct the original state for evaluation.
    """
    def __init__(self, num_latent: int, num_trash: int = 2,
                 backend=None, shots: int = 1024):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.backend = backend or Aer.get_backend('qasm_simulator')
        self.shots = shots
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, 'q')
        cr = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(qr, cr)

        # Encode the latent vector with a variational ansatz
        qc.compose(RealAmplitudes(self.num_latent, reps=3), range(self.num_latent), inplace=True)
        qc.barrier()

        # Swap‑test with the trash qubits
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def compress(self, latent_vector: np.ndarray) -> float:
        """
        Apply the autoencoder circuit to the given latent vector and return
        the expectation value of the auxiliary qubit (swap‑test success).
        """
        qc = self._circuit.copy()
        # Apply the latent vector as a unitary rotation (simplified)
        for idx, val in enumerate(latent_vector):
            qc.rx(val, idx)
        result = execute(qc, self.backend, shots=self.shots, memory=True).result()
        counts = result.get_counts()
        exp_z = sum((1 if bit == '0' else -1) * count for bit, count in counts.items()) / self.shots
        return exp_z

    def reconstruct(self, latent_vector: np.ndarray) -> Statevector:
        """
        Invert the autoencoder circuit to reconstruct the original state.
        """
        inv = self._circuit.inverse()
        qc = QuantumCircuit(self.num_latent + 2 * self.num_trash + 1)
        qc.compose(inv, inplace=True)
        # Apply the latent vector
        for idx, val in enumerate(latent_vector):
            qc.rx(val, idx)
        sv = Statevector.from_label('0' * (self.num_latent + 2 * self.num_trash + 1))
        return sv.evolve(qc)

    def get_classifier_circuit(self, num_qubits: int, depth: int):
        """
        Provide a simple variational classifier circuit that can operate on the
        reconstructed state.  Mirrors the build_classifier_circuit from the
        quantum reference pair.
        """
        from qiskit.circuit import ParameterVector
        encoding = ParameterVector('x', num_qubits)
        weights = ParameterVector('theta', num_qubits * depth)
        qc = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            qc.rx(param, qubit)
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        observables = [SparsePauliOp('I' * i + 'Z' + 'I' * (num_qubits - i - 1))
                       for i in range(num_qubits)]
        return qc, list(encoding), list(weights), observables
