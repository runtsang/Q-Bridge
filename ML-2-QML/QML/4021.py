import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

class SamplerQNN:
    """Quantum implementation of SamplerQNN that embeds both a variational sampler
    and a self‑attention style block in a single circuit."""
    def __init__(self, n_qubits: int = 4, input_dim: int = 2):
        self.n_qubits = n_qubits
        self.input_dim = input_dim

        # Parameter vectors
        self.input_params = ParameterVector("input", input_dim)
        self.weight_params = ParameterVector("weight", 4)
        self.rotation_params = ParameterVector("rot", n_qubits * 3)
        self.entangle_params = ParameterVector("ent", n_qubits - 1)

        # Backend
        self.backend = Aer.get_backend("qasm_simulator")

    def _attention_circuit(self, rot_vals, ent_vals):
        """Builds the attention sub‑circuit."""
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rot_vals[3 * i], i)
            qc.ry(rot_vals[3 * i + 1], i)
            qc.rz(rot_vals[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(ent_vals[i], i, i + 1)
        return qc

    def _sampler_circuit(self, input_vals, weight_vals):
        """Builds the sampler sub‑circuit."""
        qc = QuantumCircuit(self.n_qubits)
        for i, val in enumerate(input_vals):
            qc.ry(val, i)
        qc.cx(0, 1)
        for i, val in enumerate(weight_vals):
            qc.ry(val, i)
        qc.cx(0, 1)
        return qc

    def run(self, input_vals: np.ndarray, weight_vals: np.ndarray,
            rotation_vals: np.ndarray, entangle_vals: np.ndarray,
            shots: int = 1024):
        """
        Execute the combined circuit on a simulator.

        Parameters
        ----------
        input_vals : array of shape (input_dim,)
        weight_vals : array of shape (4,)
        rotation_vals : array of shape (n_qubits*3,)
        entangle_vals : array of shape (n_qubits-1,)

        Returns
        -------
        counts : dict mapping bitstring → count
        """
        qc = QuantumCircuit(self.n_qubits)
        qc += self._attention_circuit(rotation_vals, entangle_vals)
        qc += self._sampler_circuit(input_vals, weight_vals)

        job = execute(qc, self.backend, shots=shots)
        result = job.result()
        return result.get_counts(qc)

__all__ = ["SamplerQNN"]
