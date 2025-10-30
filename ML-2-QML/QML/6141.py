"""
Quantum convolution filter (quanvolution) that emulates the classical Conv filter.
The circuit is parameterised and uses a short random depth to keep the hardware cost low.
"""

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit import execute

class QuanvCircuit:
    """
    Quantum filter that processes a 2×2 patch and returns the average probability
    of measuring |1⟩ across all qubits.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square patch (kernel_size × kernel_size).
    backend : qiskit.providers.Backend, optional
        Backend used for execution; defaults to Aer qasm_simulator.
    shots : int, default 100
        Number of shots for the measurement.
    threshold : float, default 0.5
        Threshold for binarising pixel values before binding to parameters.
    """
    def __init__(self, kernel_size=2, backend=None, shots=100, threshold=0.5):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        qc = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, p in enumerate(self.theta):
            qc.rx(p, i)
        qc.barrier()
        qc += random_circuit(self.n_qubits, depth=2)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a single 2×2 patch.

        Parameters
        ----------
        data : np.ndarray
            Shape (kernel_size, kernel_size). Values are expected in [0,1].

        Returns
        -------
        float
            Average probability of measuring |1⟩ per qubit.
        """
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data_flat:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)

        # Compute mean number of |1⟩ outcomes
        total_ones = 0
        for bitstring, count in result.items():
            ones = sum(int(bit) for bit in bitstring)
            total_ones += ones * count

        return total_ones / (self.shots * self.n_qubits)

__all__ = ["QuanvCircuit"]
