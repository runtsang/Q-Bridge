import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

# Quantum self‑attention‑enhanced variational circuit
class HybridEstimatorQNNQuantum:
    """
    Builds a variational quantum circuit with an attention‑style entangling layer
    and wraps it in a qiskit_machine_learning EstimatorQNN for regression.
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("statevector_simulator")
        self.estimator = StatevectorEstimator()

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray,
                       data: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        # Data encoding with Ry rotations
        for i in range(self.n_qubits):
            qc.ry(data[i], i)
        # Parameterised rotation layer
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        # Attention‑style entangling: controlled‑rx between neighbours
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        return qc

    def build_estimator(self,
                        rotation_params: np.ndarray,
                        entangle_params: np.ndarray) -> EstimatorQNN:
        # Create a circuit template (parameterised only by data)
        circuit_template = QuantumCircuit(self.qr, self.cr)
        # Data‑dependent part will be added during execution
        # Define observable (Pauli‑Y on all qubits)
        observable = SparsePauliOp.from_list([("Y" * self.n_qubits, 1)])
        # Instantiate EstimatorQNN with the statevector estimator
        estimator_qnn = EstimatorQNN(
            circuit=circuit_template,
            observables=observable,
            input_params=[],  # No input params; data is encoded directly
            weight_params=rotation_params,
            estimator=self.estimator,
        )
        return estimator_qnn

    def run(self,
            data: np.ndarray,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> float:
        # Build the full circuit with data encoding
        qc = self._build_circuit(rotation_params, entangle_params, data)
        # Evaluate the expectation value of the observable
        obs = SparsePauliOp.from_list([("Y" * self.n_qubits, 1)])
        result = self.estimator.run(backend=self.backend,
                                    circuits=qc,
                                    observables=obs,
                                    parameter_values={})
        return result[0].data.values[0].real

__all__ = ["HybridEstimatorQNNQuantum"]
