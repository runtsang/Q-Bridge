import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.random import random_circuit
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter

class SelfAttentionHybrid:
    """
    Quantum implementation that mirrors the classical SelfAttentionHybrid.
    It consists of:
      - A convolution‑style circuit that encodes each token into a qubit register.
      - A variational attention circuit with rotation and entanglement gates.
    """
    def __init__(self, n_qubits: int = 4, filter_size: int = 2) -> None:
        self.n_qubits = n_qubits
        self.filter_size = filter_size
        self.backend = AerSimulator()
        self._conv_circuit = self._build_conv_circuit()
        self._attention_template = self._build_attention_template()

    def _build_conv_circuit(self) -> QuantumCircuit:
        n = self.filter_size ** 2
        qr = QuantumRegister(n, 'q')
        cr = ClassicalRegister(n, 'c')
        qc = QuantumCircuit(qr, cr)
        theta = [Parameter(f'theta{i}') for i in range(n)]
        for i in range(n):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(n, 2)
        qc.measure_all()
        return qc

    def _build_attention_template(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(Parameter(f'rot{3*i}'), i)
            qc.ry(Parameter(f'rot{3*i+1}'), i)
            qc.rz(Parameter(f'rot{3*i+2}'), i)
        for i in range(self.n_qubits - 1):
            qc.crx(Parameter(f'ent{ i }'), i, i + 1)
        qc.barrier()
        qc.measure_all()
        return qc

    def run(
        self,
        inputs: Sequence[np.ndarray],
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """
        Execute the hybrid quantum pipeline.
        Parameters
        ----------
        inputs : list of 2D arrays of shape (filter_size, filter_size)
            Each entry represents a token to be encoded via the conv circuit.
        rotation_params : np.ndarray
            Array of 3 * n_qubits rotation angles for the attention circuit.
        entangle_params : np.ndarray
            Array of entanglement angles; length n_qubits-1.
        shots : int
            Number of shots per circuit.
        Returns
        -------
        dict
            Contains the average |1> probability of the conv circuit for each
            token and the raw counts from the attention circuit.
        """
        # 1. Convolution circuit for each token
        conv_results = []
        for data in inputs:
            n = self.filter_size ** 2
            param_binds = []
            for val in data.flatten():
                # Bind each theta to π if the classical value exceeds 0.5
                bind = {self._conv_circuit.parameters[0]: np.pi if val > 0.5 else 0.0}
                param_binds.append(bind)
            job = qiskit.execute(
                self._conv_circuit,
                self.backend,
                shots=shots,
                parameter_binds=param_binds,
            )
            result = job.result().get_counts(self._conv_circuit)
            # average probability of measuring |1> across all qubits
            total = 0
            for bits, count in result.items():
                ones = sum(int(b) for b in bits)
                total += ones * count
            conv_results.append(total / (shots * n))

        # 2. Attention circuit
        attn_qc = self._attention_template.copy()
        # Bind rotation parameters
        for i in range(self.n_qubits):
            attn_qc.assign_parameters(
                {f'rot{3*i}': rotation_params[3*i], f'rot{3*i+1}': rotation_params[3*i+1],
                 f'rot{3*i+2}': rotation_params[3*i+2]}, inplace=True
            )
        # Bind entanglement parameters
        for i in range(self.n_qubits - 1):
            attn_qc.assign_parameters({f'ent{i}': entangle_params[i]}, inplace=True)

        job = qiskit.execute(attn_qc, self.backend, shots=shots)
        attn_counts = job.result().get_counts(attn_qc)

        return {"conv_probs": conv_results, "attention_counts": attn_counts}
