import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridConvQCNN:
    """
    Quantum implementation that merges the Conv filter logic with a QCNN‑style ansatz.
    The circuit first encodes the 8‑dimensional input via a ZFeatureMap, then applies
    a stack of convolution and pooling blocks identical to the classical QCNN helper,
    and finally measures an observable that returns a single expectation value.
    """
    def __init__(self, backend=None, shots: int = 2048, threshold: float = 127.0) -> None:
        self.backend = backend if backend is not None else Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        # Feature map for data encoding
        self.feature_map = ZFeatureMap(8)

        # Build the ansatz
        self.circuit = self._build_circuit()

        # Observable for a single qubit expectation measurement
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        # EstimatorQNN for evaluating the circuit
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    # ------------------------------------------------------------------
    # Convolution and pooling primitives (identical to the QCNN reference)
    # ------------------------------------------------------------------
    def _conv_circuit(self, params):
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

    def _conv_layer(self, num_qubits, prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        params = ParameterVector(prefix, length=num_qubits * 3)
        for idx in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[idx * 3:(idx + 2) * 3])
            qc.append(sub, [qubits[idx], qubits[idx + 1]])
            qc.barrier()
        return qc

    def _pool_circuit(self, params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources, sinks, prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(prefix, length=len(sources) * 3)
        for src, sink, idx in zip(sources, sinks, range(len(sources))):
            sub = self._pool_circuit(params[idx * 3:(idx + 1) * 3])
            qc.append(sub, [src, sink])
            qc.barrier()
        return qc

    def _build_circuit(self):
        # Start with the feature map
        circuit = QuantumCircuit(8)
        circuit.append(self.feature_map, range(8))

        # Convolution–pooling stages
        ansatz = QuantumCircuit(8)

        # First conv + pool
        ansatz.compose(self._conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

        # Second conv + pool
        ansatz.compose(self._conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

        # Third conv + pool
        ansatz.compose(self._conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        # Combine
        circuit.compose(ansatz, range(8), inplace=True)
        return circuit

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def run(self, data: np.ndarray) -> float:
        """
        Execute the QNN on a single 8‑dimensional input vector.

        Parameters
        ----------
        data : np.ndarray
            Input data of shape (8,).  Values are interpreted as angles for the
            ZFeatureMap; a simple thresholding step can be applied beforehand
            to emulate the classical Conv filter logic.

        Returns
        -------
        float
            Expectation value of the observable after the variational circuit.
        """
        # Ensure the data shape matches the feature map
        if data.ndim!= 1 or data.size!= 8:
            raise ValueError("Input data must be a 1‑D array of length 8.")
        result = self.qnn.predict(data.reshape(1, -1))
        return float(result[0])

__all__ = ["HybridConvQCNN"]
