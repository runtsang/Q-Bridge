import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes

class SelfAttentionAutoencoder:
    """
    Quantum hybrid module that first builds a self‑attention style circuit
    (parameter‑ised rotations + controlled‑X entanglement) and then
    feeds the resulting state into a variational auto‑encoder circuit.
    The design follows the SelfAttention.py and Autoencoder.py quantum
    seeds but integrates them into a single callable object.
    """

    def __init__(self, n_qubits: int, latent_qubits: int = 3, trash_qubits: int = 2):
        self.n_qubits = n_qubits
        self.latent_qubits = latent_qubits
        self.trash_qubits = trash_qubits
        self.backend = Aer.get_backend("qasm_simulator")

    def _rotation_circuit(self, rot_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rot_params[3 * i], i)
            qc.ry(rot_params[3 * i + 1], i)
            qc.rz(rot_params[3 * i + 2], i)
        return qc

    def _entangle_circuit(self, ent_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits - 1):
            qc.crx(ent_params[i], i, i + 1)
        return qc

    def _autoencoder_circuit(self, latent: int, trash: int) -> QuantumCircuit:
        qr = QuantumRegister(latent + 2 * trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        ansatz = RealAmplitudes(latent + trash, reps=5)
        qc.compose(ansatz, range(0, latent + trash), inplace=True)
        qc.barrier()
        aux = latent + 2 * trash
        qc.h(aux)
        for i in range(trash):
            qc.cswap(aux, latent + i, latent + trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """
        Execute the full hybrid circuit: rotation → entanglement → auto‑encoder.
        Parameters are flattened arrays of length 3*n_qubits and n_qubits‑1
        respectively.
        """
        # Build self‑attention part
        qc = QuantumCircuit(self.n_qubits)
        qc += self._rotation_circuit(rotation_params)
        qc += self._entangle_circuit(entangle_params)

        # Append auto‑encoder part
        ae_qc = self._autoencoder_circuit(self.latent_qubits, self.trash_qubits)
        qc.append(ae_qc, range(self.n_qubits))

        # Measurement
        qc.measure_all()
        job = execute(qc, self.backend, shots=shots)
        return job.result().get_counts(qc)

__all__ = ["SelfAttentionAutoencoder"]
