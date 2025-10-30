import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridQuantumNetwork:
    """
    Quantum counterpart of HybridFCLQCNN.  Builds a variational circuit that
    concatenates a ZFeatureMap, a QCNN‑style ansatz (convolution + pooling),
    and a fully‑connected quantum layer consisting of parameterised Ry gates.
    The circuit is wrapped in an EstimatorQNN so that the same ``run`` method
    can be called with a flat list of parameters.
    """

    def __init__(self, n_qubits: int = 8, shots: int = 512) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.estimator = Estimator()
        self.qnn = self._build_qnn()

    # --------------------------------------------------------------------- #
    #   Convolution and pooling primitives (QCNN style)
    # --------------------------------------------------------------------- #
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

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(param_vec[i:i+3])
            qc.compose(sub, [i, i+1], inplace=True)
            qc.barrier()
        return qc

    def _pool_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._pool_circuit(param_vec[i:i+3])
            qc.compose(sub, [i, i+1], inplace=True)
            qc.barrier()
        return qc

    # --------------------------------------------------------------------- #
    #   Build the full ansatz
    # --------------------------------------------------------------------- #
    def _build_ansatz(self) -> QuantumCircuit:
        # First convolution & pooling
        qc = QuantumCircuit(self.n_qubits)
        qc.compose(self._conv_layer(self.n_qubits, "c1"), inplace=True)
        qc.compose(self._pool_layer(self.n_qubits, "p1"), inplace=True)

        # Second stage on reduced qubits
        sub_qc = QuantumCircuit(self.n_qubits // 2)
        sub_qc.compose(self._conv_layer(self.n_qubits // 2, "c2"), inplace=True)
        sub_qc.compose(self._pool_layer(self.n_qubits // 2, "p2"), inplace=True)
        qc.compose(sub_qc, list(range(self.n_qubits // 2, self.n_qubits)), inplace=True)

        # Third stage on even smaller set
        sub_qc = QuantumCircuit(self.n_qubits // 4)
        sub_qc.compose(self._conv_layer(self.n_qubits // 4, "c3"), inplace=True)
        sub_qc.compose(self._pool_layer(self.n_qubits // 4, "p3"), inplace=True)
        qc.compose(sub_qc, list(range(self.n_qubits // 4, self.n_qubits // 2)), inplace=True)

        return qc

    # --------------------------------------------------------------------- #
    #   Assemble the full QNN
    # --------------------------------------------------------------------- #
    def _build_qnn(self) -> EstimatorQNN:
        feature_map = ZFeatureMap(self.n_qubits)
        ansatz = self._build_ansatz()

        # Fully‑connected quantum layer (Ry rotations)
        fc_params = ParameterVector("fc", length=self.n_qubits)
        fc_qc = QuantumCircuit(self.n_qubits)
        for i, param in enumerate(fc_params):
            fc_qc.ry(param, i)

        # Full circuit
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        circuit.compose(fc_qc, inplace=True)

        # Observable on the first qubit
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)])

        return EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters + list(fc_params),
            estimator=self.estimator,
        )

    # --------------------------------------------------------------------- #
    #   Public API
    # --------------------------------------------------------------------- #
    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Predicts the expectation value for a flat list of parameters.
        The ordering must match the concatenated list of:
          * feature_map parameters
          * ansatz parameters (conv/pool)
          * fully‑connected quantum layer parameters
        """
        return self.qnn.predict(thetas)


def HybridQuantumNetworkFactory() -> HybridQuantumNetwork:
    """Return a ready‑to‑use instance of the hybrid quantum network."""
    return HybridQuantumNetwork()


__all__ = ["HybridQuantumNetwork", "HybridQuantumNetworkFactory"]
