import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.providers.aer import AerSimulator

class QuantumAttentionBlock:
    """Parameterised self‑attention block used by :class:`HybridConv`."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)

    def build(self, rotation_params: ParameterVector, entangle_params: ParameterVector) -> QuantumCircuit:
        for i in range(self.n_qubits):
            self.circuit.rx(rotation_params[3*i], i)
            self.circuit.ry(rotation_params[3*i+1], i)
            self.circuit.rz(rotation_params[3*i+2], i)
        for i in range(self.n_qubits-1):
            self.circuit.crx(entangle_params[i], i, i+1)
        self.circuit.measure_all()
        return self.circuit

class HybridConv:
    """Hybrid quantum convolutional block that fuses a quanvolution
    sub‑circuit, a QCNN‑style ansatz and a self‑attention unit.  The output
    is the average probability of measuring |1> across all qubits,
    optionally scaled with fraud‑detection style parameters."""
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127,
        attention_qubits: int = 4,
        shots: int = 1024,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.attention_qubits = attention_qubits
        self.shots = shots
        self.n_qubits = kernel_size ** 2 + attention_qubits
        self.backend = AerSimulator()
        self.scale = 1.0
        self.shift = 0.0
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        # Quanvolution part – simple RX rotations per data qubit
        for i in range(self.kernel_size ** 2):
            theta = Parameter(f"theta{i}")
            qc.rx(theta, i)
        qc.barrier()
        # Attention part – use the QuantumAttentionBlock
        attn = QuantumAttentionBlock(self.attention_qubits)
        rotation_params = ParameterVector("rot", 3 * self.attention_qubits)
        entangle_params = ParameterVector("ent", self.attention_qubits - 1)
        attn_circ = attn.build(rotation_params, entangle_params)
        qc.append(attn_circ.to_instruction(), range(self.kernel_size ** 2, self.n_qubits))
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """Execute the circuit on a 2‑D array of pixel intensities.

        Parameters
        ----------
        data
            Array of shape (kernel_size, kernel_size) with values in [0, 255].
        """
        bind = {}
        # Bind parameters for the quanvolution part
        for i, val in enumerate(data.reshape(-1)):
            bind[f"theta{i}"] = np.pi if val > self.threshold else 0
        # Random rotations for the attention block
        for i in range(self.attention_qubits):
            bind[f"rot_{3*i}"] = np.random.uniform(0, 2*np.pi)
            bind[f"rot_{3*i+1}"] = np.random.uniform(0, 2*np.pi)
            bind[f"rot_{3*i+2}"] = np.random.uniform(0, 2*np.pi)
        for i in range(self.attention_qubits - 1):
            bind[f"ent_{i}"] = np.random.uniform(0, 2*np.pi)
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind])
        result = job.result().get_counts(self.circuit)
        total_ones = sum(int(bit) for key in result for bit in key)
        prob = total_ones / (self.shots * self.n_qubits)
        # fraud‑style scaling
        return prob * self.scale + self.shift

__all__ = ["HybridConv"]
