import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

from.FastBaseEstimator import FastBaseEstimator
from.SelfAttention import SelfAttention

class ConvHybridQuantum:
    """Hybrid quantum convolution + self‑attention block with fast expectation evaluation."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, n_qubits: int = 4):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = n_qubits
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.conv_circuit = self._build_conv_circuit()
        self.attention = SelfAttention(n_qubits=n_qubits)
        # FastBaseEstimator will evaluate expectation values of the conv circuit
        self.estimator = FastBaseEstimator(self.conv_circuit)

    def _build_conv_circuit(self) -> QuantumCircuit:
        n = self.kernel_size ** 2
        qc = QuantumCircuit(n)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n)]
        for i in range(n):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += qiskit.circuit.random.random_circuit(n, 2)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum convolution circuit on a single data point.
        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).
        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data = np.reshape(data, (1, self.kernel_size ** 2))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.conv_circuit.parameters[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(self.conv_circuit,
                             self.backend,
                             shots=100,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self.conv_circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        prob = counts / (100 * self.kernel_size ** 2)

        # Simple quantum self‑attention on the probability vector
        rot = np.random.randn(self.attention.n_qubits, self.attention.n_qubits)
        ent = np.random.randn(self.attention.n_qubits, self.attention.n_qubits)
        counts_att = self.attention.run(self.backend, rot, ent, shots=1024)
        # Convert counts to a mean probability of measuring |1> in the attention block
        total_counts = sum(counts_att.values())
        ones_att = sum(int(k) * v for k, v in counts_att.items())
        return ones_att / (total_counts * self.attention.n_qubits)

    def evaluate(self,
                 observables,
                 parameter_sets):
        """
        Evaluate expectation values for a list of observables and parameter sets.
        """
        return self.estimator.evaluate(observables, parameter_sets)

def Conv() -> ConvHybridQuantum:
    """Convenience factory matching the original Conv.py interface."""
    return ConvHybridQuantum()

__all__ = ["ConvHybridQuantum", "Conv"]
