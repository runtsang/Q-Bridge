"""Quantum implementation of a convolutional filter.

The class mimics the original ``Conv`` factory but uses a
parameter‑ised Qiskit circuit.  It can be dropped into the
ML code as a stand‑in for the quantum filter.  The circuit
contains a single RX rotation per qubit whose angle is a
trainable parameter.  The output is the average probability
of measuring |1> across all qubits, which is differentiable
via the parameter‑shift rule if the backend supports it.
"""

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.providers.aer import AerSimulator

class ConvIntegrator:
    """
    Quantum convolutional filter that can be used as a drop‑in
    replacement for the classical Conv class in the original
    project.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        shots: int = 1024,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size * kernel_size
        self.shots = shots

        # Parameterised circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        # RX rotations with trainable angles
        for i, p in enumerate(self.theta):
            self.circuit.rx(p, i)
        # Simple entangling layer
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

        self.backend = AerSimulator()
        self.job = None

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a single 2‑D data array.

        Parameters
        ----------
        data : np.ndarray
            Shape (kernel_size, kernel_size).  Values are
            interpreted as thresholded binary features.
        Returns
        -------
        float
            Average probability of measuring |1> over all qubits.
        """
        # Flatten data and threshold
        flat = data.flatten()
        bound = {p: np.pi if v > self.threshold else 0.0 for p, v in zip(self.theta, flat)}

        # Execute
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bound],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute average |1> probability
        total_ones = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq
        prob = total_ones / (self.shots * self.n_qubits)
        return prob

__all__ = ["ConvIntegrator"]
