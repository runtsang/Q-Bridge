from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
import numpy as np

class QuantumCombinedSamplerAttentionQNN:
    """
    Variational quantum sampler with an embedded attention subcircuit.
    The attention block uses rotation and entanglement parameters,
    followed by a sampler consisting of Ry/CX gates.
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.attn_rot = ParameterVector("attn_rot", 3 * n_qubits)
        self.attn_ent = ParameterVector("attn_ent", n_qubits - 1)
        self.sample_ry = ParameterVector("sample_ry", 2 * n_qubits)

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        # Attention rotation gates
        for i in range(self.n_qubits):
            qc.rx(self.attn_rot[3 * i], i)
            qc.ry(self.attn_rot[3 * i + 1], i)
            qc.rz(self.attn_rot[3 * i + 2], i)
        # Attention entanglement CRX
        for i in range(self.n_qubits - 1):
            qc.crx(self.attn_ent[i], i, i + 1)
        # Sampler Ry gates
        for i in range(self.n_qubits):
            qc.ry(self.sample_ry[i], i)
        qc.cx(0, 1)
        for i in range(self.n_qubits):
            qc.ry(self.sample_ry[self.n_qubits + i], i)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self, backend=None, shots=1024,
            attn_rot_vals=None, attn_ent_vals=None, sample_ry_vals=None):
        if backend is None:
            backend = AerSimulator()
        qc = self._build_circuit()
        param_dict = {}
        if attn_rot_vals is not None:
            param_dict.update({self.attn_rot[i]: v for i, v in enumerate(attn_rot_vals)})
        if attn_ent_vals is not None:
            param_dict.update({self.attn_ent[i]: v for i, v in enumerate(attn_ent_vals)})
        if sample_ry_vals is not None:
            param_dict.update({self.sample_ry[i]: v for i, v in enumerate(sample_ry_vals)})
        bound_qc = qc.bind_parameters(param_dict)
        job = execute(bound_qc, backend, shots=shots)
        return job.result().get_counts(bound_qc)

__all__ = ["QuantumCombinedSamplerAttentionQNN"]
