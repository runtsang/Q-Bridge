import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

class ConvGen404:
    """Hybrid‑compatible convolution filter with quantum implementation.

    The quantum circuit encodes the pixel values into rotation angles and
    measures the expectation of the Z operator across all qubits.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 backend=None, shots: int = 100) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, t in enumerate(theta):
            qc.rx(t, i)
        qc.barrier()
        qc += random_circuit(self.n_qubits, 2)
        qc.measure_all()
        self.theta = theta
        return qc

    def run(self, data: np.ndarray) -> float:
        """Execute the quantum filter.

        Parameters
        ----------
        data : np.ndarray
            2D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1>.
        """
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for row in data_flat:
            bind = {t: np.pi if val > self.threshold else 0
                    for t, val in zip(self.theta, row)}
            param_binds.append(bind)

        job = qiskit.execute(self._circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

    @staticmethod
    def sampler_qnn():
        """Return a simple parameterized quantum sampler circuit."""
        from qiskit.circuit import ParameterVector
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = qiskit.QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        return qc

    @staticmethod
    def estimator_qnn():
        """Return a simple parameterized quantum estimator circuit."""
        from qiskit.circuit import Parameter
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp
        param1 = Parameter("input1")
        param2 = Parameter("weight1")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(param1, 0)
        qc.rx(param2, 0)
        observable = SparsePauliOp.from_list([("Y", 1)])
        return qc, observable

__all__ = ["ConvGen404"]
