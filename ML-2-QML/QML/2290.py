"""
Quantum hybrid module: a parameterised circuit that implements an
autoencoder‑style swap test and a fully‑connected layer via rotation angles.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler

class HybridQuantumCircuit:
    """Parameterised quantum circuit that couples an autoencoder ansatz with a swap test."""

    def __init__(self, num_latent: int, num_trash: int, backend, shots: int = 1024):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.backend = backend
        self.shots = shots
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        total_qubits = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Autoencoder ansatz on latent + trash qubits
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=5)
        qc.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)

        qc.barrier()

        # Swap test with auxiliary qubit
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters and return the
        expectation value of the auxiliary qubit.
        """
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[
                {self.circuit.parameters[i]: theta} for i, theta in enumerate(thetas)
            ],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        outcomes = np.array([int(k) for k in counts.keys()])
        expectation = np.sum(outcomes * probs)
        return np.array([expectation])


def HybridQuantumCircuitFactory(
    num_latent: int,
    num_trash: int,
    shots: int = 1024,
) -> HybridQuantumCircuit:
    """Convenience factory mirroring the original QML helper."""
    backend = qiskit.Aer.get_backend("qasm_simulator")
    return HybridQuantumCircuit(num_latent, num_trash, backend, shots)


__all__ = [
    "HybridQuantumCircuit",
    "HybridQuantumCircuitFactory",
]
