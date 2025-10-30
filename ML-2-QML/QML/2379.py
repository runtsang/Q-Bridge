"""HybridSamplerQNN: Quantum module combining a variational sampler with a classical filter."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN


class HybridSamplerQNNQuantum:
    """
    A quantum sampler that encodes a 2×2 image patch into a parameterized circuit.
    The circuit uses RX rotations conditioned on the pixel values and a random
    entangling layer. The output is the average probability of measuring |1⟩
    across all qubits. The class is compatible with Qiskit Machine Learning
    interfaces for gradient‑based training.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 100,
        threshold: float = 127,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold

        # Backend and sampler
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.sampler = StatevectorSampler(self.backend)

        # Parameter vectors for inputs and weights
        self.inputs = ParameterVector("input", self.n_qubits)
        self.weights = ParameterVector("weight", self.n_qubits)

        # Build the variational circuit
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self._circuit.rx(self.inputs[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        # Wrap into a Qiskit ML SamplerQNN object
        self.sampler_qnn = QSamplerQNN(
            circuit=self._circuit,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=self.sampler,
        )

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a single 2×2 image patch.

        Args:
            data: 2D array of shape (kernel_size, kernel_size) with integer pixel values.
        Returns:
            Average probability of measuring |1⟩ across all qubits.
        """
        flat_data = np.reshape(data, (1, self.n_qubits))

        # Bind input parameters based on threshold
        param_binds = []
        for dat in flat_data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.inputs[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        # Compute average probability of |1⟩
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

    def sample(self, data: np.ndarray) -> np.ndarray:
        """
        Return a probability distribution over the two possible outcomes
        of the sampler network, using the quantum circuit as a feature extractor.
        """
        prob = self.run(data)
        # Convert to a 2‑element probability vector
        return np.array([1.0 - prob, prob])

__all__ = ["HybridSamplerQNNQuantum"]
