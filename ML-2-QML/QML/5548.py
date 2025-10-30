from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit import Aer
from qiskit_machine_learning.neural_networks import SamplerQNN

# --------------------------------------------------------------------------- #
# Quantum Self‑attention block with autoencoder‑style feature extraction
# --------------------------------------------------------------------------- #
class SelfAttention:
    """
    Quantum self‑attention that builds a parameterised circuit for the
    query/key projections, composes a simple autoencoder sub‑circuit,
    and samples the result.  The interface matches the classical
    counterpart so the two can be swapped during experiments.
    """
    def __init__(self, n_qubits: int = 4, latent_dim: int = 3, trash: int = 2):
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        self.trash = trash
        self.backend = Aer.get_backend("qasm_simulator")

    # --------------------------------------------------------------------- #
    # 1. Build the attention circuit
    # --------------------------------------------------------------------- #
    def _build_attention_circuit(self, rotation_params: np.ndarray,
                                 entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        return qc

    # --------------------------------------------------------------------- #
    # 2. Autoencoder sub‑circuit (borrowed from the classical design)
    # --------------------------------------------------------------------- #
    def _build_autoencoder_circuit(self) -> QuantumCircuit:
        num_latent = self.latent_dim
        num_trash  = self.trash
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Parameterised ansatz for the latent sub‑space
        qc.compose(RealAmplitudes(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        qc.barrier()

        # Domain‑wall: flip the second half of the trash qubits
        for i in range(num_trash, 2 * num_trash):
            qc.x(i)

        # Swap‑test style measurement
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    # --------------------------------------------------------------------- #
    # 3. Run the composite circuit
    # --------------------------------------------------------------------- #
    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> dict[str, int]:
        """
        Execute the attention + autoencoder circuit and return measurement counts.
        """
        attention_circ = self._build_attention_circuit(rotation_params, entangle_params)
        ae_circ = self._build_autoencoder_circuit()
        composite = attention_circ.compose(ae_circ, inplace=False)

        job = self.backend.run(composite, shots=shots)
        result = job.result()
        return result.get_counts(composite)

def SelfAttention() -> SelfAttention:
    """Factory mirroring the classical helper."""
    return SelfAttention(n_qubits=4)

__all__ = ["SelfAttention"]
