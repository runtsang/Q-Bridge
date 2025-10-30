import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from typing import Iterable

class HybridConvFC:
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 127,
                 n_features: int = 1,
                 shots: int = 100) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.threshold = threshold
        self.shots = shots

        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self._weight = np.random.randn(n_features + 1)  # +1 for conv output
        self._bias = np.random.randn()

    def run(self, data: np.ndarray, thetas: Iterable[float]) -> float:
        """Return a scalar output.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) representing
            the image patch to be processed.
        thetas : Iterable[float]
            External parameters that are fed into the linear read‑out.

        Returns
        -------
        float
            The result of the hybrid quantum‑classical layer.
        """
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data_flat:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
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
        conv_out = counts / (self.shots * self.n_qubits)

        feature = np.concatenate(([conv_out], np.array(list(thetas), dtype=np.float32)))
        expectation = np.tanh(np.dot(self._weight, feature) + self._bias)
        return float(expectation)

__all__ = ["HybridConvFC"]
