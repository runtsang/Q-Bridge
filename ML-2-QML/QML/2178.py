"""
Quantum filter with a 400‑pixel kernel implemented as a parameter‑efficient variational circuit.

The quantum version keeps the same public interface as the classical Conv:
- ``run`` receives a 2‑D NumPy array and returns a scalar.
- The 400×400 patch is encoded into a set of rotation angles.
- A shallow variational circuit (single‑qubit rotations + a single layer of CX gates) reuses the same angles for all qubits, reducing parameter count.
- The measurement returns the average probability of measuring |1> across all qubits.

The circuit is executed on a Qiskit Aer simulator but can be swapped with any Qiskit backend.
"""

import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit import Aer, execute

def Conv():
    class QuanvCircuit:
        """Quantum filter for a 400×400 patch using a parameter‑efficient circuit."""

        def __init__(self, kernel_size: int = 400, backend=None, shots: int = 1000, threshold: float = 127):
            self.kernel_size = kernel_size
            self.n_qubits = kernel_size ** 2
            self.backend = backend or Aer.get_backend("qasm_simulator")
            self.shots = shots
            self.threshold = threshold

            # 400×400 angles are encoded as a single rotation angle per qubit
            # We reuse the same rotation gate for all qubits to keep parameter count low
            self.theta = ParameterVector("theta", self.n_qubits)
            self.circuit = qiskit.QuantumCircuit(self.n_qubits, self.n_qubits)

            # Encode data into rotation angles
            for i in range(self.n_qubits):
                self.circuit.ry(self.theta[i], i)

            # Variational layer: one layer of CX gates in a checkerboard pattern
            for i in range(0, self.n_qubits - 1, 2):
                self.circuit.cx(i, i + 1)
            for i in range(1, self.n_qubits - 1, 2):
                self.circuit.cx(i, i + 1)

            # Measurement
            self.circuit.measure(range(self.n_qubits), range(self.n_qubits))

        def run(self, data: np.ndarray) -> float:
            """
            Execute the quantum filter on a 2‑D array.

            Parameters
            ----------
            data : np.ndarray
                2‑D array with shape (H, W) containing pixel values in [0, 255].

            Returns
            -------
            float
                Average probability of measuring |1> across all qubits.
            """
            # Flatten and binarize the data using the threshold
            flat = np.reshape(data, (self.n_qubits,))
            bind_dict = {self.theta[i]: (np.pi if val > self.threshold else 0) for i, val in enumerate(flat)}

            # Execute circuit
            job = execute(self.circuit, backend=self.backend, shots=self.shots,
                          parameter_binds=[bind_dict])
            result = job.result().get_counts(self.circuit)

            # Compute average |1> probability
            total_ones = 0
            for bitstring, count in result.items():
                total_ones += sum(int(b) for b in bitstring) * count

            return total_ones / (self.shots * self.n_qubits)

    return QuanvCircuit()
