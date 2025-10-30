import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import L_BFGS_B
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

class QCNNHybrid:
    """
    Quantum Convolutional Neural Network implemented with Qiskit.
    Supports configurable depth, qubit count and training via the
    EstimatorQNN interface.  The architecture mirrors the classical
    QCNNHybrid but replaces each fully‑connected block with a
    parameterised two‑qubit convolution and a pooling operation.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        depth: int = 3,
        param_init: float | None = None,
        seed: int | None = 1234,
    ):
        self.n_qubits = n_qubits
        self.depth = depth
        self.seed = seed
        self.estimator = Estimator()
        self.feature_map = ZFeatureMap(n_qubits)
        self._build_circuit(param_init)

    # ------------------------------------------------------------------
    # Sub‑circuit helpers (convolution & pooling)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Layer constructors
    # ------------------------------------------------------------------
    def _conv_layer(self, layer_idx: int) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        param_vec = ParameterVector(f"c{layer_idx}", 3 * (self.n_qubits // 2))
        idx = 0
        for q1, q2 in zip(range(0, self.n_qubits, 2), range(1, self.n_qubits, 2)):
            sub = self._conv_circuit(param_vec[idx : idx + 3])
            qc.compose(sub, [q1, q2], inplace=True)
            qc.barrier()
            idx += 3
        return qc

    def _pool_layer(self, layer_idx: int, sources: list[int]) -> QuantumCircuit:
        sinks = [s + 1 for s in sources]
        qc = QuantumCircuit(self.n_qubits)
        param_vec = ParameterVector(f"p{layer_idx}", 3 * len(sources))
        idx = 0
        for src, sink in zip(sources, sinks):
            sub = self._pool_circuit(param_vec[idx : idx + 3])
            qc.compose(sub, [src, sink], inplace=True)
            qc.barrier()
            idx += 3
        return qc

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------
    def _build_circuit(self, param_init: float | None):
        self.circuit = QuantumCircuit(self.n_qubits)
        # Feature map
        self.circuit.compose(self.feature_map, range(self.n_qubits), inplace=True)
        # Build layers
        for d in range(self.depth):
            self.circuit.compose(self._conv_layer(d), range(self.n_qubits), inplace=True)
            # After convolution, shrink the active qubits
            active = list(range(self.n_qubits // 2 ** (d + 1)))
            self.circuit.compose(self._pool_layer(d, active), range(self.n_qubits), inplace=True)
        # Observation on first qubit
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )
        if param_init is not None:
            init_vals = np.full(len(self.circuit.parameters), param_init)
            self.qnn.set_weights(init_vals)

    # ------------------------------------------------------------------
    # Training & inference
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        maxiter: int = 200,
        **kwargs,
    ):
        """
        Train the QCNN using L_BFGS_B optimisation.
        """
        clf = NeuralNetworkClassifier(
            qnn=self.qnn,
            optimizer=L_BFGS_B(maxiter=maxiter, **kwargs),
            training_dataset={"x": X, "y": y},
            target_accuracy=0.99,
        )
        clf.run()
        # Persist the trained parameters
        self.trained_params = clf.get_optimal_parameters()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return binary predictions (0/1) for the given samples.
        """
        preds = self.qnn.predict(X)
        return (preds >= 0.5).astype(int)
