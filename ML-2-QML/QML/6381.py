"""Hybrid quantum self‑attention with a quanvolution filter.

The circuit first prepares a quanvolution layer that maps a 2‑D patch to
a qubit register.  Afterwards a variational block parameterised by
rotation_params and entangle_params implements the attention logic.
The API is identical to the original SelfAttention module so that
classical and quantum versions can be swapped.
"""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.random import random_circuit

class HybridSelfAttentionQuantum:
    def __init__(self, kernel_size: int = 2, n_qubits: int = 4, shots: int = 1024):
        self.kernel_size = kernel_size
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        # Quanvolution part
        self.quanv = self._build_quanv()

    def _build_quanv(self):
        """Create a parameterised quanvolution circuit."""
        qc = QuantumCircuit(self.n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, p in enumerate(theta):
            qc.rx(p, i)
        qc.barrier()
        qc += random_circuit(self.n_qubits, depth=2)
        qc.measure_all()
        return qc

    def _build_attention(self, rotation_params, entangle_params):
        """Variational attention block."""
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            data: np.ndarray,
            shots: int | None = None) -> dict:
        """
        Args:
            rotation_params: array of length 3*n_qubits
            entangle_params: array of length n_qubits-1
            data: 2‑D patch of shape (kernel_size, kernel_size)
        Returns:
            dict of counts from the attention circuit
        """
        # Prepare data‑dependent parameters for quanvolution
        data_flat = np.reshape(data, (self.n_qubits,))
        param_binds = []
        for val in data_flat:
            bind = {f"theta{i}": np.pi if val > 0.5 else 0 for i in range(self.n_qubits)}
            param_binds.append(bind)

        # Execute quanvolution
        quanv_job = qiskit.execute(self.quanv,
                                   self.backend,
                                   shots=self.shots,
                                   parameter_binds=param_binds)
        quanv_counts = quanv_job.result().get_counts(self.quanv)

        # Build and execute attention block
        attn_qc = self._build_attention(rotation_params, entangle_params)
        attn_job = qiskit.execute(attn_qc,
                                  self.backend,
                                  shots=self.shots)
        attn_counts = attn_job.result().get_counts(attn_qc)

        return {"quanv": quanv_counts, "attention": attn_counts}

def SelfAttention():
    """Return a quantum hybrid self‑attention object."""
    return HybridSelfAttentionQuantum(kernel_size=2, n_qubits=4, shots=512)

__all__ = ["SelfAttention"]
