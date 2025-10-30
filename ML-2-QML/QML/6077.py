from qiskit.circuit import ParameterVector
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, execute
import numpy as np

class HybridSamplerAttention:
    """
    Quantum hybrid sampler‑attention module.
    Builds a composite circuit that first applies a parameterised sampler
    on two qubits, then a self‑attention style block on four qubits.
    The circuit is executed on a QASM simulator and returns measurement
    counts for the attention qubits.
    """

    def __init__(self):
        # Parameter vectors
        self.input_params = ParameterVector("input", 2)
        self.sampler_weights = ParameterVector("sampler_weight", 4)
        self.attn_rot = ParameterVector("rot", 12)   # 3 params per qubit
        self.attn_ent = ParameterVector("ent", 3)    # 3 entangling gates

        # Sampler circuit (2 qubits)
        qr_s = QuantumRegister(2, "s")
        sampler_circ = QuantumCircuit(qr_s)
        sampler_circ.ry(self.input_params[0], 0)
        sampler_circ.ry(self.input_params[1], 1)
        sampler_circ.cx(0, 1)
        for i in range(4):
            sampler_circ.ry(self.sampler_weights[i], i % 2)

        # Attention circuit (4 qubits)
        qr_a = QuantumRegister(4, "a")
        attn_circ = QuantumCircuit(qr_a)
        for i in range(4):
            idx = 3 * i
            attn_circ.rx(self.attn_rot[idx], i)
            attn_circ.ry(self.attn_rot[idx + 1], i)
            attn_circ.rz(self.attn_rot[idx + 2], i)
        for i in range(3):
            attn_circ.crx(self.attn_ent[i], i, i + 1)

        # Combined circuit
        cr = ClassicalRegister(4, "c")
        self.combined = QuantumCircuit(qr_s, qr_a, cr)
        self.combined.compose(sampler_circ, qubits=qr_s, inplace=True)
        self.combined.compose(attn_circ, qubits=qr_a, inplace=True)
        self.combined.measure(qr_a, cr)

        # Backend
        self.backend = Aer.get_backend("qasm_simulator")

    def run(self, shots: int = 1024):
        """
        Execute the composite circuit and return measurement counts.
        """
        job = execute(self.combined, self.backend, shots=shots)
        return job.result().get_counts(self.combined)

__all__ = ["HybridSamplerAttention"]
