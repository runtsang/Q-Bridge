"""HybridQuantumCircuit: Variational circuit for attention weighting."""

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2
from qiskit import Aer, execute

class HybridQuantumCircuit:
    """
    A parameterised quantum circuit that generates a data‑dependent
    attention vector.  The circuit is built from an EfficientSU2 ansatz
    and a simple data‑encoding layer that maps each pixel to a rotation
    angle via a threshold.  The output is the expectation value of Z
    on each qubit, averaged to produce a scalar attention weight
    per channel.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.BaseBackend = None,
        shots: int = 100,
        threshold: float = 0.5,
    ):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        if backend is None:
            self.backend = Aer.get_backend("qasm_simulator")
        else:
            self.backend = backend

        # Data‑encoding circuit: each qubit gets an RX gate with angle
        # set to 0 or π depending on the pixel value.
        self.theta = [Parameter(f"θ{i}") for i in range(self.n_qubits)]
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()

        # Add a variational ansatz (EfficientSU2) to increase expressivity.
        self.ansatz = EfficientSU2(self.n_qubits, reps=1)
        self._circuit.append(self.ansatz, self._circuit.qubits)

        # Measure all qubits in Z basis
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the circuit on a single kernel patch.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size) with
                  values in [0,1].

        Returns:
            np.ndarray: 1‑D array of length `n_qubits` containing the
            expectation value of Z for each qubit (values in [-1,1]).
        """
        # Flatten and reshape to (1, n_qubits)
        flat = data.reshape(1, self.n_qubits)

        # Bind parameters based on threshold
        param_binds = []
        for sample in flat:
            bind = {}
            for i, val in enumerate(sample):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        # Execute the circuit
        job = execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self._circuit)

        # Compute expectation value of Z for each qubit
        exp_z = np.zeros(self.n_qubits)
        total_shots = self.shots * len(param_binds)
        for bitstring, cnt in counts.items():
            # bitstring is in little‑endian order
            for q, bit in enumerate(bitstring[::-1]):
                exp_z[q] += ((-1) ** int(bit)) * cnt
        exp_z /= total_shots

        return exp_z

    def get_attention(self, data: np.ndarray) -> np.ndarray:
        """
        Convert Z expectation values to a scalar attention weight per channel.
        The mapping is sigmoid( (exp_z + 1) / 2 ) to bring values into [0,1].

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size).

        Returns:
            np.ndarray: 1‑D array of length `kernel_size` containing a
            scalar attention weight per channel.  For a single‑channel
            input this is a single float.
        """
        exp_z = self.run(data)
        # Collapse to a single weight per channel (here we just average)
        weight = np.mean((exp_z + 1) / 2)
        return np.array([weight])

__all__ = ["HybridQuantumCircuit"]
