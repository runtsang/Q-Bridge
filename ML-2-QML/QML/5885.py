"""Hybrid quantum self‑attention with a variational autoencoder encoder."""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class HybridSelfAttentionQML:
    """Quantum hybrid attention: variational autoencoder + swap‑test attention."""
    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.backend = StatevectorSampler()
        self.auto_circuit = self._build_autoencoder()
        self.att_circuit = self._build_attention()

    def _build_autoencoder(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        # Variational autoencoder ansatz
        qc.append(RealAmplitudes(self.n_qubits, reps=3), qr)
        # Swap‑test for reconstruction fidelity
        qc.h(qr[0])
        for i in range(1, self.n_qubits):
            qc.cswap(qr[0], qr[i], qr[i])
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        return qc

    def _build_attention(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        # Simple attention‑style entanglement
        for i in range(self.n_qubits - 1):
            qc.cx(qr[i], qr[i + 1])
        qc.measure(qr[self.n_qubits - 1], cr[0])
        return qc

    def run(self, shots: int = 1024) -> dict:
        """Execute the composite circuit and return outcome counts."""
        composite = self.auto_circuit + self.att_circuit
        job = self.backend.run(composite, shots=shots)
        return job.result().get_counts()

def HybridSelfAttentionQMLFactory(n_qubits: int = 8) -> HybridSelfAttentionQML:
    return HybridSelfAttentionQML(n_qubits)

__all__ = ["HybridSelfAttentionQML", "HybridSelfAttentionQMLFactory"]
