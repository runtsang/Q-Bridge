"""ConvGen – quantum convolution filter.

This class implements a variational quantum circuit that mimics the
behaviour of the classical filter.  The circuit encodes each pixel
into a rotation gate, applies a random entangling layer and measures
all qubits.  The returned value is the average probability of
measuring |1> across all qubits, after applying a threshold
parameter on the pixel values.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit.random import random_circuit

__all__ = ["ConvGen"]

class ConvGen:
    """
    Quantum convolution filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel; the number of qubits equals
        ``kernel_size**2``.
    threshold : float, default 0.0
        Pixel values above this threshold are encoded as a π rotation.
    backend : qiskit.backends.Backend, optional
        Quantum backend.  If ``None`` a default Aer simulator is used.
    shots : int, default 1024
        Number of shots per evaluation.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        backend=None,
        shots: int = 1024,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Build the circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"θ{i}") for i in range(self.n_qubits)]
        for i, param in enumerate(self.theta):
            self.circuit.rx(param, i)
        self.circuit.barrier()
        # Random entangling layer
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray | list | tuple) -> float:
        """
        Run the quantum filter on a single data sample.

        Parameters
        ----------
        data : array-like
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        flat = data.flatten()
        bind = {
            theta: np.pi if val > self.threshold else 0.0
            for theta, val in zip(self.theta, flat)
        }
        param_binds = [bind]

        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute average number of |1> across all qubits
        total_ones = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq

        return total_ones / (self.shots * self.n_qubits)
