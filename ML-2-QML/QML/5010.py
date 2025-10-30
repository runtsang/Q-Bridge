from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
import numpy as np
from typing import List, Sequence, Tuple

class UnifiedClassifier:
    """Quantum classifier using a data‑uploading ansatz and EstimatorQNN."""
    def __init__(self, num_qubits: int, depth: int = 3, shots: int | None = None, seed: int | None = None) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.seed = seed
        self.circuit, self.encoding_params, self.weight_params, self.observables = self._build_circuit()
        estimator = StatevectorEstimator()
        self.estimator = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.encoding_params,
            weight_params=self.weight_params,
            estimator=estimator,
        )
        # Store weight sizes for compatibility with the classical counterpart
        self.weight_sizes = [len(self.weight_params)]

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        qc = QuantumCircuit(self.num_qubits)
        # Feature encoding
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)
        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        # Observables used as “output” qubits
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1)) for i in range(self.num_qubits)]
        return qc, list(encoding), list(weights), observables

    def evaluate(self, parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """
        Evaluate the quantum circuit using EstimatorQNN.
        Mirrors the FastEstimator noise model when shots are specified.
        """
        results = self.estimator.evaluate(self.observables, parameter_sets)
        if self.shots is None:
            return results
        rng = np.random.default_rng(self.seed)
        noisy = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    @staticmethod
    def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthetic dataset for quantum regression/classification.
        Mirrors the data generation logic from the regression reference pair.
        """
        omega_0 = np.zeros(2 ** num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_wires, dtype=complex)
        omega_1[-1] = 1.0
        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2 ** num_wires), dtype=complex)
        for i in range(samples):
            states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        labels = np.sin(2 * thetas) * np.cos(phis)
        return states, labels
