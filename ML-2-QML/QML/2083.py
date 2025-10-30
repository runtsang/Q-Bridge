"""
ConvFusion: Quantum variational filter for hybrid convolution.
"""

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal
from qiskit import Aer, execute

class ConvFusion:
    """
    Quantum filter that implements a parameterized variational circuit
    over a grid of qubits equal to kernel_size^2.

    The circuit applies data‑dependent RX rotations followed by an
    entangling layer. The measurement probability of |1> is returned
    as the quantum activation.
    """

    def __init__(self,
                 kernel_size: int,
                 backend: str = "qasm_simulator",
                 shots: int = 1024,
                 threshold: float = 0.0):
        self.n_qubits = kernel_size ** 2
        self.kernel_size = kernel_size
        self.shots = shots
        self.threshold = threshold
        self.backend = Aer.get_backend(backend)

        # Parameterized circuit
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.params = [Parameter(f"θ{idx}") for idx in range(self.n_qubits)]

        # RX rotations with data‑dependent angles
        for q, θ in zip(range(self.n_qubits), self.params):
            self.circuit.rx(θ, q)

        # Entangling layer
        self.circuit.append(
            TwoLocal(self.n_qubits,
                     reps=2,
                     rotation_blocks="ry",
                     entanglement="circular",
                     entanglement_blocks="cx"),
            range(self.n_qubits)
        )

        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on classical data.

        Args:
            data: 2D array of shape (kernel_size, kernel_size) with values in [0, 1].

        Returns:
            float: average probability of measuring |1> across all qubits.
        """
        # Map data to angles: values > threshold -> π else 0
        angles = np.pi * (data > self.threshold).astype(float).flatten()

        # Bind parameters
        param_binds = [{p: θ for p, θ in zip(self.params, angles)}]

        job = execute(self.circuit,
                      backend=self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)

        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute probability of |1> for each qubit
        total_ones = 0
        for bitstring, freq in counts.items():
            # bitstring is in reverse order
            ones = sum(int(bit) for bit in bitstring)
            total_ones += ones * freq

        prob = total_ones / (self.shots * self.n_qubits)
        return prob
