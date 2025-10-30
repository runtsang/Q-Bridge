import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN

# ------------------------------------------------------------------
# 1. Quantum self‑attention (inspired by SelfAttention.py)
# ------------------------------------------------------------------
class QuantumSelfAttention:
    """
    Parameterised quantum circuit that emulates a self‑attention block.
    The circuit uses RX/RZ rotations for query/key/value and a chain of
    entangling CX gates to mix the qubits.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # Rotation layer
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entangling chain
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
            circuit.cx(i + 1, i)
        # Measurement
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


# ------------------------------------------------------------------
# 2. Quantum fraud‑detection style block (simplified)
# ------------------------------------------------------------------
class QuantumFraudDetection:
    """
    A small variational circuit that mimics the photonic fraud‑detection
    circuit.  It uses single‑qubit rotations followed by a CX chain and
    final measurement.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(params[3 * i], i)
            circuit.ry(params[3 * i + 1], i)
            circuit.rz(params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, params: np.ndarray, shots: int = 1024) -> dict:
        circuit = self._build_circuit(params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


# ------------------------------------------------------------------
# 3. Quantum auto‑encoder (from Autoencoder.py)
# ------------------------------------------------------------------
class QuantumAutoencoder:
    """
    Variational auto‑encoder implemented with a RealAmplitudes ansatz
    and a SamplerQNN.  The circuit uses a swap‑test style measurement
    to obtain a scalar output.
    """
    def __init__(self, num_latent: int = 3, num_trash: int = 2, backend=None) -> None:
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        self.cr = ClassicalRegister(1, "c")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # Ansatz
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=5)
        circuit.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)
        circuit.barrier()
        # Swap‑test
        aux = self.num_latent + 2 * self.num_trash
        circuit.h(aux)
        for i in range(self.num_trash):
            circuit.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, self.cr[0])
        return circuit

    def run(self, shots: int = 1024) -> dict:
        job = qiskit.execute(self.circuit, self.backend, shots=shots)
        return job.result().get_counts(self.circuit)


# ------------------------------------------------------------------
# 4. Hybrid quantum pipeline
# ------------------------------------------------------------------
class HybridSelfAttentionModel:
    """
    Quantum counterpart of the classical HybridSelfAttentionModel.
    The same class name is used for a direct mapping between the two
    implementations.
    """
    def __init__(self, backend=None) -> None:
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.attention = QuantumSelfAttention(n_qubits=4)
        self.fraud = QuantumFraudDetection(n_qubits=4)
        self.autoencoder = QuantumAutoencoder(num_latent=3, num_trash=2, backend=self.backend)

    def run_quantum(self, inputs: np.ndarray, shots: int = 1024) -> dict:
        """
        Run the full quantum pipeline.  The input is a 1‑D array that is
        embedded into rotation parameters for the attention and fraud blocks.
        """
        # Pad or truncate to match expected parameter length
        size = self.attention.n_qubits * 3
        params = np.pad(inputs, (0, size - len(inputs)), mode="constant")
        rotation_params = params[:size]
        entangle_params = params[size:2 * size]

        attn_counts = self.attention.run(self.backend, rotation_params, entangle_params, shots)
        fraud_counts = self.fraud.run(self.backend, params[:size], shots)
        ae_counts = self.autoencoder.run(shots)

        return {
            "attention": attn_counts,
            "fraud": fraud_counts,
            "autoencoder": ae_counts,
        }

__all__ = ["HybridSelfAttentionModel"]
