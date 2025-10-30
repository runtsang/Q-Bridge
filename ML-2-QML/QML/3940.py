"""Quantum implementation of a simple autoencoder layer."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler

class HybridAutoencoder:
    """Quantum implementation of a simple autoencoder layer."""
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qr = QuantumRegister(self.n_qubits)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        qc.compose(RealAmplitudes(self.n_qubits, reps=3), qr, inplace=True)
        qc.measure(qr[0], cr[0])
        return qc

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Evaluate the circuit for a batch of parameters."""
        binds = [{self.circuit.parameters[i]: th for i, th in enumerate(thetas)}]
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array([counts.get(bit,0) for bit in ["0","1"]], dtype=float) / self.shots
        expectation = probs[1] - probs[0]
        return np.array([expectation])

def HybridAutoencoderFactory(
    n_qubits: int = 4,
    *,
    backend=None,
    shots: int = 1024,
) -> HybridAutoencoder:
    return HybridAutoencoder(n_qubits, backend, shots)

__all__ = ["HybridAutoencoder", "HybridAutoencoderFactory"]
