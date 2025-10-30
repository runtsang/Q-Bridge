"""Quantum circuit implementing a QCNN‑style ansatz enriched with a
randomised layer inspired by Quantum‑NAT.

The model exposes a callable EstimatorQNN that can be used directly
with qiskit‑machine‑learning optimisers.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RandomCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QCNNHybrid:
    """
    Hybrid QCNN‑style quantum neural network.

    The ansatz contains:
        * A feature‑map layer (ZFeatureMap) that encodes the classical input.
        * Three convolutional layers built from a custom 2‑qubit unitary
          (the same as in the QCNN reference).
        * Two pooling layers that collapse pairs of qubits.
        * A randomised circuit layer (RandomCircuit) inserted between the
          second convolution and the first pooling to inject non‑trivial
          entanglement, mirroring the RandomLayer of Quantum‑NAT.
    """
    def __init__(self, num_qubits: int = 8, seed: int = 12345) -> None:
        self.num_qubits = num_qubits
        self.seed = seed
        self.estimator = StatevectorEstimator()
        self.qnn = self._build_qnn()

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[i // 2 * 3 : i // 2 * 3 + 3])
            qc.append(sub, [i, i + 1])
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources, sinks, prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=len(sources) * 3)
        for src, snk, i in zip(sources, sinks, range(len(sources))):
            sub = self._pool_circuit(params[i * 3 : i * 3 + 3])
            qc.append(sub, [src, snk])
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        ansatz = QuantumCircuit(self.num_qubits)
        # First convolutional layer
        ansatz.compose(self._conv_layer(self.num_qubits, "c1"), inplace=True)
        # Randomised layer (Quantum‑NAT style)
        rand = RandomCircuit(num_qubits=self.num_qubits, depth=2, seed=self.seed)
        ansatz.compose(rand, inplace=True)
        # First pooling layer
        ansatz.compose(self._pool_layer(list(range(0, self.num_qubits, 2)),
                                        list(range(1, self.num_qubits, 2)),
                                        "p1"), inplace=True)
        # Second convolutional layer
        ansatz.compose(self._conv_layer(self.num_qubits // 2, "c2"), inplace=True)
        # Second pooling layer
        ansatz.compose(self._pool_layer(list(range(0, self.num_qubits // 2, 2)),
                                        list(range(1, self.num_qubits // 2, 2)),
                                        "p2"), inplace=True)
        return ansatz

    def _build_qnn(self) -> EstimatorQNN:
        feature_map = ZFeatureMap(num_qubits=self.num_qubits)
        ansatz = self._build_ansatz()
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])
        qnn = EstimatorQNN(
            circuit=ansatz.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator,
        )
        return qnn

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the QNN on a batch of classical inputs."""
        return self.qnn(inputs)

    def parameters(self):
        """Return the trainable parameters of the QNN."""
        return self.qnn.weight_params

__all__ = ["QCNNHybrid"]
