import numpy as np
import qiskit
from qiskit import Aer
from qiskit.circuit.random import random_circuit

class ConvFilter:
    """
    Quantum‑based convolutional filter with an adaptive variational circuit.
    The circuit contains a trainable parameter vector and a threshold that
    is supplied by a classical controller.  The class exposes a run method
    compatible with the original seed API.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 1024,
        threshold: float = 0.5,
        backend=None,
    ):
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit(kernel_size)

    def _build_circuit(self, kernel_size: int):
        n = kernel_size ** 2
        circ = qiskit.QuantumCircuit(n)
        # Parameterised rotation angles
        theta = qiskit.circuit.ParameterVector("theta", length=n)
        for i in range(n):
            circ.rx(theta[i], i)
        # Entanglement layer
        for i in range(n - 1):
            circ.cx(i, i + 1)
        circ.barrier()
        circ += random_circuit(n, 2)
        circ.measure_all()
        return circ

    def run(self, data):
        """
        Run the quantum circuit on the supplied 2‑D patch.

        Parameters
        ----------
        data : array‑like, shape (kernel_size, kernel_size)
            Classical input that determines the parameter binding.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data_flat:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.circuit.parameters[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)

        total = 0
        for key, cnt in result.items():
            ones = sum(int(bit) for bit in key)
            total += ones * cnt

        return total / (self.shots * self.n_qubits)

    def set_threshold(self, new_threshold: float):
        """Update the classical threshold used for parameter binding."""
        self.threshold = new_threshold

__all__ = ["ConvFilter"]
