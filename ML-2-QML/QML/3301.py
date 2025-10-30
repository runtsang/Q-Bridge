import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler

class HybridAttentionAutoencoder:
    """Quantum model combining a self‑attention circuit with a swap‑test autoencoder."""
    def __init__(self,
                 n_qubits: int = 4,
                 latent_dim: int = 3,
                 trash: int = 2):
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        self.trash = trash
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.sampler = Sampler()

    def _self_attention_circuit(self, rot_params, ent_params):
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rot_params[3 * i], i)
            qc.ry(rot_params[3 * i + 1], i)
            qc.rz(rot_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(ent_params[i], i, i + 1)
        return qc

    def _autoencoder_circuit(self, latent_params, trash_params):
        qr = QuantumRegister(self.latent_dim + 2 * self.trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        ansatz = RealAmplitudes(self.latent_dim + self.trash, reps=5)
        qc.compose(ansatz, [j for j in range(self.latent_dim + self.trash)], inplace=True)
        qc.barrier()
        aux = self.latent_dim + 2 * self.trash
        qc.h(aux)
        for i in range(self.trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def run(self, rot_params, ent_params, latent_params, trash_params, shots=1024):
        sa_circ = self._self_attention_circuit(rot_params, ent_params)
        ae_circ = self._autoencoder_circuit(latent_params, trash_params)
        qc = sa_circ + ae_circ
        job = self.sampler.run(qc, shots=shots)
        return job.result().get_counts(qc)

def SelfAttention(n_qubits: int = 4,
                  latent_dim: int = 3,
                  trash: int = 2):
    """Factory mirroring the classical helper, returning a configured quantum hybrid."""
    return HybridAttentionAutoencoder(n_qubits, latent_dim, trash)

__all__ = ["SelfAttention", "HybridAttentionAutoencoder"]
