"""Quantum autoencoder with integrated self‑attention block.

The circuit combines a variational encoder (RealAmplitudes) with a
self‑attention sub‑circuit built from RX/RY/RZ gates and controlled‑RZ
entangling operations.  A swap‑test extracts a latent register that
serves as the compressed representation.  The implementation uses
Qiskit’s SamplerQNN to interface with a classical optimiser if needed."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class AutoencoderHybrid:
    """Quantum variational autoencoder with a self‑attention block."""
    def __init__(
        self,
        latent_dim: int = 3,
        trash_dim: int = 2,
        attention_qubits: int = 4,
        reps: int = 5,
    ) -> None:
        algorithm_globals.random_seed = 42
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.attention_qubits = attention_qubits
        self.reps = reps

        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        # Build the full circuit once; parameters will be updated during optimisation
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=(2,),
        )

    # ----------------------------------------------------------------------
    def _attention_subcirc(self, qr: QuantumRegister, offset: int) -> QuantumCircuit:
        """Self‑attention style sub‑circuit using single‑qubit rotations and
        controlled‑RZ gates.  The pattern mirrors the classical attention
        from the reference but in a quantum‑gate form."""
        circ = QuantumCircuit(qr)
        for i in range(self.attention_qubits):
            circ.rx(np.random.rand(), offset + i)
            circ.ry(np.random.rand(), offset + i)
            circ.rz(np.random.rand(), offset + i)
        for i in range(self.attention_qubits - 1):
            circ.crz(np.random.rand(), offset + i, offset + i + 1)
        return circ

    def _build_circuit(self) -> QuantumCircuit:
        total_qubits = self.latent_dim + 2 * self.trash_dim + 1  # +1 for ancilla
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Variational encoder
        encoder = RealAmplitudes(
            num_qubits=self.latent_dim + self.trash_dim,
            reps=self.reps,
        )
        circuit.append(encoder, range(self.latent_dim + self.trash_dim))

        # Self‑attention block acting on all qubits except ancilla
        attention_circ = self._attention_subcirc(qr, 0)
        circuit.append(attention_circ, list(range(total_qubits - 1)))

        # Swap test with ancilla
        ancilla = self.latent_dim + 2 * self.trash_dim
        circuit.h(ancilla)
        for i in range(self.trash_dim):
            circuit.cswap(ancilla,
                         self.latent_dim + i,
                         self.latent_dim + self.trash_dim + i)
        circuit.h(ancilla)
        circuit.measure(ancilla, cr[0])
        return circuit

    # ----------------------------------------------------------------------
    def encode(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """Run the circuit and return the measurement counts for the ancilla."""
        job = qiskit.execute(self.circuit, self.backend, shots=shots)
        return job.result().get_counts(self.circuit)

    def run(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """Convenience wrapper that forwards to encode."""
        return self.encode(inputs, shots=shots)

__all__ = ["AutoencoderHybrid"]
