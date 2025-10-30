import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from typing import Iterable

class HybridFCConvLayer:
    """
    Quantum implementation of a hybrid fully‑connected + convolutional layer.
    The circuit first encodes a 2×2 data patch into a set of RX gates, applies
    a random entangling block, and finally a parameterised Ry on a dedicated
    output qubit.  The expectation value of all qubits is returned as the
    layer output.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 127,
                 backend=None, shots: int = 100) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Build the quantum circuit
        self.circuit = qiskit.QuantumCircuit(self.n_qubits + 1)
        self.theta = qiskit.circuit.Parameter("theta")
        self.theta_params = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]

        # Encode data with RX gates
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta_params[i], i)

        # Entangling block
        self.circuit += random_circuit(self.n_qubits, 2)

        # Fully‑connected qubit (qubit 0)
        self.circuit.ry(self.theta, 0)
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, data: np.ndarray, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit on a 2×2 data patch.

        Parameters
        ----------
        data
            2‑D array of shape (kernel_size, kernel_size) with values in [0, 255].
        thetas
            Iterable containing a single parameter for the Ry gate.

        Returns
        -------
        np.ndarray
            Array containing the average expectation value of all qubits.
        """
        param_binds = []
        for val in data.flatten():
            bind = {}
            for i, theta_param in enumerate(self.theta_params):
                bind[theta_param] = np.pi if val > self.threshold else 0
            bind[self.theta] = thetas[0]
            param_binds.append(bind)

        job = qiskit.execute(self.circuit, self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        expectation = counts / (self.shots * (self.n_qubits + 1))
        return np.array([expectation])

__all__ = ["HybridFCConvLayer"]
