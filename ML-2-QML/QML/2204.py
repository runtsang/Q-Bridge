import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit.primitives import StatevectorEstimator


class QuanvCircuit:
    """
    Variational quanvolution circuit that replaces the classical
    ConvFilter.  The circuit prepares a state on n_qubits = kernel_size²
    using RX rotations parameterized by the input pixel values, then
    applies a random two‑qubit layer.  Measurement is performed on all
    qubits and the average |1⟩ probability is returned as the feature.
    """
    def __init__(self, kernel_size: int, backend, shots: int, threshold: int):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]

        # Encode pixel values into RX rotations
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()

        # Add a small random entangling layer
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a single 2×2 patch.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape (kernel_size, kernel_size) with pixel
            intensities in the range [0, 255].

        Returns
        -------
        float
            Average probability of measuring |1⟩ across all qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)


def EstimatorQNN() -> QEstimatorQNN:
    """
    Construct the quantum EstimatorQNN.  The quanvolution circuit
    serves as the feature extractor; a StatevectorEstimator evaluates
    the expectation value of a Y⊗…⊗Y observable.  The resulting
    scalar is the output of the quantum neural network.
    """
    backend = qiskit.Aer.get_backend("qasm_simulator")
    filter_size = 2
    qc = QuanvCircuit(filter_size, backend, shots=100, threshold=127)

    observable = SparsePauliOp.from_list([("Y" * qc.n_qubits, 1)])
    estimator = StatevectorEstimator()

    estimator_qnn = QEstimatorQNN(
        circuit=qc._circuit,
        observables=observable,
        input_params=[qc.theta[0]],
        weight_params=[qc.theta[1]],
        estimator=estimator,
    )
    return estimator_qnn


__all__ = ["EstimatorQNN"]
