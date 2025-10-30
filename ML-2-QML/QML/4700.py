import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

class HybridLayer:
    """
    Quantum counterpart to the classical HybridLayer. Builds a parameterized
    circuit that mirrors the convolutional filter and fully connected
    transformation of the classical layer and returns a single expectation
    value. The circuit uses a data-dependent rotation for the convolution
    part and a free parameter set for the fully connected part.
    """

    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 127):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        # Convolutional part: RX gates parameterized by data
        self.theta_conv = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta_conv[i], i)
        self._circuit.barrier()
        # Random entangling layer to increase expressivity
        self._circuit += random_circuit(self.n_qubits, 2)
        # Fully connected part: free parameters
        self.theta_fcl = [qiskit.circuit.Parameter(f"phi{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.ry(self.theta_fcl[i], i)
        self._circuit.barrier()
        self._circuit.measure_all()
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray, thetas: Sequence[float]) -> np.ndarray:
        """
        Execute the hybrid quantum circuit on the given data and parameters.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape (kernel_size, kernel_size) representing the
            convolutional input.
        thetas : Sequence[float]
            Parameters for the fully connected part of the circuit.

        Returns
        -------
        np.ndarray
            Array containing a single expectation value (average |1> probability).
        """
        # Flatten data and bind convolutional parameters
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data_flat:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta_conv[i]] = np.pi if val > self.threshold else 0
            bind.update({self.theta_fcl[i]: thetas[i] for i in range(self.n_qubits)})
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)
        counts = 0
        total = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
            total += val
        expectation = counts / (total * self.n_qubits)
        return np.array([expectation])

__all__ = ["HybridLayer"]
