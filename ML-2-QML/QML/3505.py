import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Single‑qubit convolution block used in the QCNN ansatz."""
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


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Single‑qubit pooling block used in the QCNN ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Build a convolution layer by tiled 2‑qubit blocks."""
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits, 2):
        params = ParameterVector(f"{prefix}_{i//2}", length=3)
        block = conv_circuit(params)
        qc.append(block, [i, i + 1])
    return qc


def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Build a pooling layer by tiled 2‑qubit blocks."""
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits, 2):
        params = ParameterVector(f"{prefix}_{i//2}", length=3)
        block = pool_circuit(params)
        qc.append(block, [i, i + 1])
    return qc


class HybridFCL_QCNN:
    """
    Quantum counterpart of the classical HybridFCL_QCNN.
    Builds a QCNN ansatz with an additional pre‑processing rotation
    on the first qubit that mimics a fully‑connected quantum layer.
    The `run` method evaluates a Pauli‑Z expectation value for a
    supplied list of parameters.
    """
    def __init__(self) -> None:
        # Estimator backend
        self.estimator = StatevectorEstimator()
        # Feature map
        self.feature_map = ZFeatureMap(8)
        # Pre‑processing quantum layer (1‑qubit fully‑connected style)
        self.pre = QuantumCircuit(1)
        self.theta_pre = Parameter("θ_pre")
        self.pre.ry(self.theta_pre, 0)
        # QCNN ansatz on 8 qubits
        self.ansatz = QuantumCircuit(8)
        self.ansatz.compose(conv_layer(8, "c1"), inplace=True)
        self.ansatz.compose(pool_layer(8, "p1"), inplace=True)
        self.ansatz.compose(conv_layer(4, "c2"), inplace=True)
        self.ansatz.compose(pool_layer(4, "p2"), inplace=True)
        self.ansatz.compose(conv_layer(2, "c3"), inplace=True)
        self.ansatz.compose(pool_layer(2, "p3"), inplace=True)
        # Attach pre‑processing to first qubit of the ansatz
        self.ansatz.compose(self.pre, [0], inplace=True)
        # Full circuit: feature map + ansatz
        self.circuit = QuantumCircuit(8)
        self.circuit.compose(self.feature_map, range(8), inplace=True)
        self.circuit.compose(self.ansatz, range(8), inplace=True)
        # Observable
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        # QNN wrapper
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=list(self.ansatz.parameters) + [self.theta_pre],
            estimator=self.estimator,
        )

    def run(self, thetas: list[float]) -> np.ndarray:
        """
        Evaluate the expectation value for a flat list of parameters.
        The mapping is:
            * first 8 parameters → feature‑map angles (ignored: set to 0)
            * next N parameters → QCNN ansatz weights
            * last parameter   → pre‑processing rotation θ_pre
        """
        input_vals = [0.0] * len(self.feature_map.parameters)
        weight_vals = {p: v for p, v in zip(self.qnn.weight_params, thetas)}
        result = self.qnn.predict(input_vals, weight_vals)
        return np.array(result)
