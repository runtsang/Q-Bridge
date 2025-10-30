import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridAutoencoder:
    """Quantum hybrid auto‑encoder that stitches an auto‑encoder ansatz with a quantum self‑attention block."""
    def __init__(self, num_latent: int = 3, num_trash: int = 2, attention_qubits: int = 4):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.attention_qubits = attention_qubits
        self.backend = AerSimulator(method="statevector")
        self.circuit = self._build_circuit()

    # ----------------------------------------------------------------------- #
    # Auto‑encoder ansatz – taken from the Autoencoder seed
    # ----------------------------------------------------------------------- #
    def _autoencoder_layer(self, num_qubits: int, reps: int = 5) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=reps)

    # ----------------------------------------------------------------------- #
    # Quantum self‑attention block – inspired by the SelfAttention seed
    # ----------------------------------------------------------------------- #
    def _quantum_self_attention(self, n_qubits: int) -> QuantumCircuit:
        qr = QuantumRegister(n_qubits, "q")
        cr = ClassicalRegister(n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)
        # Rotation layer
        for i in range(n_qubits):
            circuit.rx(0.1 * i, i)
            circuit.ry(0.2 * i, i)
            circuit.rz(0.3 * i, i)
        # Simple entanglement
        for i in range(n_qubits - 1):
            circuit.crx(0.05 * i, i, i + 1)
        circuit.measure(qr, cr)
        return circuit

    # ----------------------------------------------------------------------- #
    # Compose the full circuit – auto‑encoder + swap‑test + self‑attention
    # ----------------------------------------------------------------------- #
    def _build_circuit(self) -> QuantumCircuit:
        total_qubits = self.num_latent + 2 * self.num_trash + 1 + self.attention_qubits
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Auto‑encoder ansatz
        ae_circ = self._autoencoder_layer(self.num_latent + self.num_trash)
        circuit.compose(ae_circ, range(0, self.num_latent + self.num_trash), inplace=True)

        # Swap‑test style measurement of latent
        aux = self.num_latent + 2 * self.num_trash
        circuit.h(aux)
        for i in range(self.num_trash):
            circuit.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])

        # Append quantum self‑attention
        att_circ = self._quantum_self_attention(self.attention_qubits)
        circuit.compose(
            att_circ,
            range(self.num_latent + 2 * self.num_trash + 1,
                  self.num_latent + 2 * self.num_trash + 1 + self.attention_qubits),
            inplace=True,
        )

        return circuit

    # ----------------------------------------------------------------------- #
    # Execution helper
    # ----------------------------------------------------------------------- #
    def run(self, shots: int = 1024) -> dict:
        """Execute the circuit on the Aer simulator and return measurement counts."""
        job = qiskit.execute(self.circuit, self.backend, shots=shots)
        return job.result().get_counts(self.circuit)

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit for inspection."""
        return self.circuit

# --------------------------------------------------------------------------- #
# Factory – mirrors the classical helper for a consistent API
# --------------------------------------------------------------------------- #
def HybridAutoencoderQML(num_latent: int = 3, num_trash: int = 2, attention_qubits: int = 4) -> HybridAutoencoder:
    """Factory that returns a quantum hybrid auto‑encoder instance."""
    return HybridAutoencoder(num_latent=num_latent, num_trash=num_trash, attention_qubits=attention_qubits)

__all__ = ["HybridAutoencoder", "HybridAutoencoderQML"]
