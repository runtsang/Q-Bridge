import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.random import random_circuit

class ConvSelfAttentionLayerQuantum:
    """Quantum counterpart: convolutional filter and self‑attention realised with Qiskit circuits."""
    def __init__(self, conv_size: int = 2, attention_qubits: int = 4,
                 shots: int = 1024, threshold: int = 127):
        self.conv_size = conv_size
        self.attention_qubits = attention_qubits
        self.shots = shots
        self.threshold = threshold
        self.backend = Aer.get_backend('qasm_simulator')
        self._conv_circuit = self._build_conv_circuit()
        self._attention_circuit = self._build_attention_circuit()

    def _build_conv_circuit(self):
        n = self.conv_size ** 2
        qc = QuantumCircuit(n)
        theta = [qiskit.circuit.Parameter(f'theta{i}') for i in range(n)]
        for i in range(n):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(n, 2)
        qc.measure_all()
        return qc

    def _build_attention_circuit(self):
        qr = QuantumRegister(self.attention_qubits, 'q')
        cr = ClassicalRegister(self.attention_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        # placeholder rotation gates (will be parametrised during run)
        for i in range(self.attention_qubits):
            qc.rx(0.0, i)
            qc.ry(0.0, i)
            qc.rz(0.0, i)
        # placeholder entanglement
        for i in range(self.attention_qubits - 1):
            qc.crx(0.0, i, i + 1)
        qc.measure(qr, cr)
        return qc

    def run(self, data: np.ndarray,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray) -> float:
        """
        Args:
            data: 2D array of shape (conv_size, conv_size) – image patch.
            rotation_params: 1‑D array of length attention_qubits – rotation angles.
            entangle_params: 1‑D array of length attention_qubits‑1 – entanglement angles.
        Returns:
            Combined probability score from conv and attention stages.
        """
        # Convolution stage
        flat = data.reshape(1, -1)
        param_bind = {f'theta{i}': np.pi if val > self.threshold else 0.0
                      for i, val in enumerate(flat[0])}
        job = execute(self._conv_circuit, self.backend,
                      shots=self.shots, parameter_binds=[param_bind])
        counts = job.result().get_counts(self._conv_circuit)
        conv_prob = sum(sum(int(bit) for bit in key) * cnt
                        for key, cnt in counts.items()) / (self.shots * self.conv_size ** 2)

        # Self‑attention stage
        qc = self._build_attention_circuit()
        for i, val in enumerate(rotation_params):
            qc.rx(val, i)
            qc.ry(val, i)
            qc.rz(val, i)
        for i, val in enumerate(entangle_params):
            qc.crx(val, i, (i + 1) % self.attention_qubits)
        job = execute(qc, self.backend, shots=self.shots)
        counts = job.result().get_counts(qc)
        attn_prob = sum(sum(int(bit) for bit in key) * cnt
                        for key, cnt in counts.items()) / (self.shots * self.attention_qubits)

        return conv_prob * attn_prob

__all__ = ["ConvSelfAttentionLayerQuantum"]
