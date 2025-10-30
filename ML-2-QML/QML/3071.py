from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as SamplerPrimitive

class ConvSamplerQNN:
    """
    Quantum hybrid convolution + sampler neural network.
    Builds a 2‑qubit variational circuit whose input
    rotations are conditioned on the first two pixels
    of the kernel, and whose internal angles are
    parameterized by a Qiskit SamplerQNN.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 shots: int = 1024,
                 backend: qiskit.providers.Backend | None = None) -> None:
        self.kernel_size = kernel_size
        self.shots = shots
        self.backend = backend or AerSimulator(method="statevector")

        # Parameter vectors
        self.inputs = ParameterVector("x", 2)
        self.weights = ParameterVector("w", 4)

        # Build the circuit
        self.circuit = QuantumCircuit(kernel_size ** 2)
        # Input rotations
        for i in range(kernel_size ** 2):
            self.circuit.ry(self.inputs[i % 2], i)

        # Entangling block
        for i in range(0, kernel_size ** 2 - 1, 2):
            self.circuit.cx(i, i + 1)

        # Weight rotations
        for i, w in enumerate(self.weights):
            self.circuit.ry(w, i % (kernel_size ** 2))

        # Measurement
        self.circuit.measure_all()

        # SamplerQNN wrapper
        self.sampler_qnn = SamplerQNN(circuit=self.circuit,
                                      input_params=self.inputs,
                                      weight_params=self.weights,
                                      sampler=SamplerPrimitive())

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum filter on a 2‑D kernel.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (kernel_size, kernel_size) with values in [0, 255].

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        flat = data.flatten()[:self.circuit.num_qubits]
        # Bind input parameters
        param_binds = [{self.inputs[i % 2]: np.pi if val > 127 else 0 for i, val in enumerate(flat)}]
        # Execute with the sampler
        result = self.sampler_qnn.sample(param_binds, shots=self.shots)
        # Compute average |1> probability
        probs = result.get_counts()
        total = self.shots * self.circuit.num_qubits
        ones = sum(sum(int(bit) for bit in key) * count for key, count in probs.items())
        return ones / total

__all__ = ["ConvSamplerQNN"]
