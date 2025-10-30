import numpy as np
import qiskit
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridFCQCNN:
    """
    Quantum counterpart of :class:`HybridFCQCNN` that implements the same
    hierarchical structure with parameterized gates:
      * a Z‑feature map acting on 8 qubits,
      * a sequence of convolutional and pooling blocks (QCNN style),
      * a single‑qubit Ry rotation acting as the FCL layer,
      * measurement of the first qubit to produce a scalar output.
    The circuit is compiled into an EstimatorQNN to allow efficient
    gradient evaluation when used as a trainable layer.
    """
    def __init__(self, shots: int = 1024) -> None:
        self.shots = shots
        self._build_circuit()
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> None:
        # Feature map
        self.feature_map = ZFeatureMap(8, reps=1, entanglement='full')

        # Convolution block
        def conv_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi/2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            qc.cx(1, 0)
            qc.rz(np.pi/2, 0)
            return qc

        # Pooling block (identical to conv but without the final rz)
        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi/2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        # Helper to compose layers over all qubit pairs
        def conv_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            qubits = list(range(num_qubits))
            params = ParameterVector(prefix, length=num_qubits//2 * 3)
            idx = 0
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                sub = conv_circuit(params[idx:idx+3])
                qc.append(sub, [q1, q2])
                qc.barrier()
                idx += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                sub = conv_circuit(params[idx:idx+3])
                qc.append(sub, [q1, q2])
                qc.barrier()
                idx += 3
            return qc

        def pool_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            qubits = list(range(num_qubits))
            params = ParameterVector(prefix, length=num_qubits//2 * 3)
            idx = 0
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                sub = pool_circuit(params[idx:idx+3])
                qc.append(sub, [q1, q2])
                qc.barrier()
                idx += 3
            return qc

        # Build the ansatz: convolution → pooling →... → FCL rotation
        self.ansatz = QuantumCircuit(8, name="Ansatz")
        self.ansatz.compose(conv_layer(8, "c1"), inplace=True)
        self.ansatz.compose(pool_layer(8, "p1"), inplace=True)
        self.ansatz.compose(conv_layer(4, "c2"), inplace=True)
        self.ansatz.compose(pool_layer(4, "p2"), inplace=True)
        self.ansatz.compose(conv_layer(2, "c3"), inplace=True)
        self.ansatz.compose(pool_layer(2, "p3"), inplace=True)
        # FCL rotation on qubit 0
        fcl_theta = qiskit.circuit.Parameter("theta_fcl")
        self.ansatz.rz(fcl_theta, 0)

        # Assemble full circuit
        self.circuit = QuantumCircuit(8)
        self.circuit.compose(self.feature_map, inplace=True)
        self.circuit.compose(self.ansatz, inplace=True)

        # Observable: Z on first qubit
        self.observable = SparsePauliOp.from_list([("Z" + "I"*7, 1)])

    def run(self, input_features: np.ndarray, weight_params: np.ndarray) -> np.ndarray:
        """
        Forward pass that evaluates the circuit.

        Parameters
        ----------
        input_features : np.ndarray
            Input vector of shape (8,), matching the ZFeatureMap dimension.
        weight_params : np.ndarray
            Array containing all variational parameters:
            [conv/pool params…, theta_fcl].
        """
        return self.qnn.predict(inputs=[input_features], weight_params=weight_params)[0]

def HybridFCQCNNFactory() -> HybridFCQCNN:
    """Factory returning a fully‑configured instance."""
    return HybridFCQCNN()

__all__ = ["HybridFCQCNN", "HybridFCQCNNFactory"]
