import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit, Aer
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class EstimatorQNNConvQuantum:
    """
    Quantum estimator that mirrors the classical hybrid model.
    A 2×2 quantum convolution circuit processes each patch,
    followed by a variational EstimatorQNN that outputs a scalar.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.Backend | None = None,
        shots: int = 1024,
        threshold: float = 0.5,
    ) -> None:
        self.kernel_size = kernel_size
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        # Build the parameterised circuit
        self.circuit = self._build_convolution_circuit()

        # Create the observable (sum of Pauli Y on all qubits)
        observable = SparsePauliOp.from_list([("Y" * self.circuit.num_qubits, 1.0)])

        # Weight parameters start random; EstimatorQNN will optimise them
        weight_params = self.circuit.parameters[self.kernel_size**2 :]

        # Instantiate the quantum neural network
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=observable,
            input_params=self.circuit.parameters[: self.kernel_size**2],
            weight_params=weight_params,
            estimator=StatevectorEstimator(),
        )

    def _build_convolution_circuit(self) -> QuantumCircuit:
        n = self.kernel_size ** 2
        qc = QuantumCircuit(n)

        # Encode input pixels with RX rotations
        rx = ParameterVector("rx", n)
        qc.rx(rx, range(n))

        # Simple entangling layer
        for i in range(n - 1):
            qc.cx(i, i + 1)

        # Randomised variational block
        qc += random_circuit(n, 2)

        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum circuit on a 2×2 patch and return the expectation value.

        Args:
            data: 2D array of shape (kernel_size, kernel_size) with values in [0, 1].

        Returns:
            float: expectation value of the observable.
        """
        # Flatten and bind parameters
        flat = data.flatten()
        param_bind = {f"rx{i}": np.pi if val > self.threshold else 0.0 for i, val in enumerate(flat)}

        job = qiskit.execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute average number of |1> outcomes weighted by counts
        total = 0
        for bitstring, cnt in counts.items():
            ones = bitstring.count("1")
            total += ones * cnt
        expectation = total / (self.shots * self.circuit.num_qubits)
        return expectation

    def run_qnn(self, data: np.ndarray) -> float:
        """
        Convenience wrapper that uses the EstimatorQNN interface.
        """
        return self.qnn.run(data)

__all__ = ["EstimatorQNNConvQuantum"]
