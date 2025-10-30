import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA

#--- quantum sub‑circuits ----------------------------------------------------
def quantum_self_attention(n_qubits: int = 4) -> QuantumCircuit:
    """Quantum implementation of a self‑attention style block."""
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(qr, cr)
    for i in range(n_qubits):
        qc.rx(np.random.rand(), i)
        qc.ry(np.random.rand(), i)
        qc.rz(np.random.rand(), i)
    for i in range(n_qubits - 1):
        qc.crx(np.random.rand(), i, i + 1)
    qc.measure(qr, cr)
    return qc


def qc_nn_ansatz(num_qubits: int = 8) -> QuantumCircuit:
    """QCNN style ansatz built from 2‑qubit convolution and pooling blocks."""
    qubits = list(range(num_qubits))
    qc = QuantumCircuit(num_qubits)
    # Convolution blocks (pairwise)
    for i in range(0, num_qubits, 2):
        qc.cx(i, i + 1)
        qc.rz(np.random.rand(), i)
        qc.ry(np.random.rand(), i + 1)
    # Pooling block (partial measurement)
    for i in range(0, num_qubits // 2):
        qc.cx(2 * i, 2 * i + 1)
        qc.rz(np.random.rand(), 2 * i)
    return qc


def quantum_nat_layer(num_wires: int = 4) -> QuantumCircuit:
    """Quantum‑NAT style random layer with parameterised single‑qubit gates."""
    qc = QuantumCircuit(num_wires)
    for w in range(num_wires):
        qc.rx(np.random.rand(), w)
        qc.ry(np.random.rand(), w)
    for pair in [(0, 1), (2, 3)]:
        qc.cx(*pair)
    return qc


#--- hybrid fraud‑detection circuit ------------------------------------------
class FraudDetectionHybrid:
    """
    Hybrid photonic‑style encoder + quantum self‑attention + QCNN + Quantum‑NAT.
    Designed to run on the Aer qasm_simulator.
    """
    def __init__(self):
        self.backend = Aer.get_backend("qasm_simulator")
        self.n_photonic_modes = 2
        self.n_qubits = 8

    def _photonic_encoder(self, inputs: np.ndarray) -> QuantumCircuit:
        """Placeholder for a photonic encoder (here, a simple rotation)."""
        qc = QuantumCircuit(self.n_photonic_modes)
        for i, val in enumerate(inputs):
            qc.rx(val, i)
        return qc

    def build_circuit(self, inputs: np.ndarray) -> QuantumCircuit:
        """Builds the full hybrid circuit from classical inputs."""
        # 1. Photonic encoder
        qc = self._photonic_encoder(inputs)

        # 2. Quantum self‑attention
        qc += quantum_self_attention(n_qubits=4)

        # 3. QCNN ansatz
        qc += qc_nn_ansatz(num_qubits=self.n_qubits)

        # 4. Quantum‑NAT layer
        qc += quantum_nat_layer(num_wires=4)

        return qc

    def run(self, inputs: np.ndarray, shots: int = 1024) -> dict:
        """Execute the hybrid circuit and return measurement counts."""
        qc = self.build_circuit(inputs)
        job = qiskit.execute(qc, self.backend, shots=shots)
        return job.result().get_counts(qc)


__all__ = ["FraudDetectionHybrid"]
