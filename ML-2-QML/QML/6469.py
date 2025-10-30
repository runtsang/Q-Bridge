from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class QuantumConvFilter:
    """Variational circuit that mimics a classical convolutional filter."""
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 256, threshold: float = 0.5) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        theta = [Parameter(f"θ{i}") for i in range(self.n_qubits)]
        for i, p in enumerate(theta):
            qc.rx(p, i)
        qc.barrier()
        qc += qiskit.circuit.random.random_circuit(self.n_qubits, depth=2)
        qc.measure_all()
        self.params = theta
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with values in [0,1].

        Returns
        -------
        float
            Estimated mean probability of measuring |1> over all qubits.
        """
        flat = data.reshape(1, self.n_qubits)
        bind = {p: np.pi if val > self.threshold else 0 for p, val in zip(self.params, flat[0])}
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind])
        result = job.result().get_counts(self.circuit)
        total = sum(count * self.shots * self.n_qubits for count in result.values())
        return total / (self.shots * self.n_qubits * len(result))

class QuantumEstimatorQNN:
    """Quantum neural network that performs regression on a single qubit."""
    def __init__(self) -> None:
        param_in = Parameter("x")
        param_wt = Parameter("w")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(param_in, 0)
        qc.rx(param_wt, 0)
        self.circuit = qc
        self.observable = SparsePauliOp.from_list([("Y", 1)])
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=qc,
            observables=self.observable,
            input_params=[param_in],
            weight_params=[param_wt],
            estimator=self.estimator,
        )

    def run(self, feature: float) -> float:
        """
        Parameters
        ----------
        feature : float
            Scalar output from the quantum convolution filter.

        Returns
        -------
        float
            Predicted regression value.
        """
        return self.qnn.predict(np.array([[feature]]))[0, 0]

class HybridConvEstimator:
    """Quantum‑enhanced version of HybridConvEstimator."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.5, shots: int = 256) -> None:
        self.conv_filter = QuantumConvFilter(kernel_size, shots=shots, threshold=threshold)
        self.estimator = QuantumEstimatorQNN()

    def run(self, patch: np.ndarray) -> float:
        """
        Parameters
        ----------
        patch : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Regression prediction.
        """
        feature = self.conv_filter.run(patch)
        return self.estimator.run(feature)

__all__ = ["HybridConvEstimator"]
