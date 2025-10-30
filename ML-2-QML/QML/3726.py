import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.providers.aer import AerSimulator

class HybridSelfAttentionCNN:
    """
    Quantum counterpart to the classical HybridSelfAttentionCNN.
    Implements the same hierarchical structure (self‑attention rotations,
    entanglement, convolution and pooling layers) as variational circuits.
    Parameters are passed at runtime via a dictionary that mirrors the
    classical interface.
    """

    def __init__(self, n_qubits: int = 8) -> None:
        self.n_qubits = n_qubits
        self.feature_map = ZFeatureMap(n_qubits, reps=1, insert_barriers=True)

        # Parameter vectors
        self.rotation_params = ParameterVector('rot', length=3 * n_qubits)
        self.entangle_params = ParameterVector('ent', length=n_qubits - 1)

        # Convolution / pooling parameters
        self.param_vectors = {}
        self.param_vectors['c1'] = ParameterVector('c1', length=n_qubits * 3)
        self.param_vectors['p1'] = ParameterVector('p1', length=(n_qubits // 2) * 3)
        self.param_vectors['c2'] = ParameterVector('c2', length=(n_qubits // 2) * 3)
        self.param_vectors['p2'] = ParameterVector('p2', length=(n_qubits // 4) * 3)
        self.param_vectors['c3'] = ParameterVector('c3', length=(n_qubits // 4) * 3)
        self.param_vectors['p3'] = ParameterVector('p3', length=(n_qubits // 8) * 3)

        # Build ansatz
        self.ansatz = self._build_ansatz()
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.compose(self.feature_map, range(n_qubits), inplace=True)
        self.circuit.compose(self.ansatz, range(n_qubits), inplace=True)

        # Estimator QNN
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
        )
        self.backend = AerSimulator()

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit convolution block with 3 parameterized rotations."""
        qc = QuantumCircuit(2)
        qc.rz(params[0], 0)
        qc.cx(1, 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit pooling block with 3 parameterized rotations."""
        qc = QuantumCircuit(2)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.rz(params[2], 0)
        return qc

    def _conv_layer(self, num_qubits: int, param_vector: ParameterVector) -> QuantumCircuit:
        """Compose convolution blocks over all adjacent qubit pairs."""
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        idx = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.append(self._conv_circuit(param_vector[idx:idx + 3]), [q1, q2])
            qc.barrier()
            idx += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.append(self._conv_circuit(param_vector[idx:idx + 3]), [q1, q2])
            qc.barrier()
            idx += 3
        return qc

    def _pool_layer(self, num_qubits: int, param_vector: ParameterVector) -> QuantumCircuit:
        """Compose pooling blocks over all adjacent qubit pairs."""
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        idx = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.append(self._pool_circuit(param_vector[idx:idx + 3]), [q1, q2])
            qc.barrier()
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Assemble the full variational ansatz."""
        qc = QuantumCircuit(self.n_qubits)
        # Self‑attention rotations
        for i in range(self.n_qubits):
            qc.rx(self.rotation_params[3 * i], i)
            qc.ry(self.rotation_params[3 * i + 1], i)
            qc.rz(self.rotation_params[3 * i + 2], i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qc.crx(self.entangle_params[i], i, i + 1)

        # Convolution / pooling hierarchy
        qc = qc.compose(self._conv_layer(self.n_qubits, self.param_vectors['c1']), list(range(self.n_qubits)), inplace=True)
        qc = qc.compose(self._pool_layer(self.n_qubits, self.param_vectors['p1']), list(range(self.n_qubits)), inplace=True)

        qc = qc.compose(self._conv_layer(self.n_qubits // 2, self.param_vectors['c2']),
                        list(range(self.n_qubits // 2, self.n_qubits)), inplace=True)
        qc = qc.compose(self._pool_layer(self.n_qubits // 2, self.param_vectors['p2']),
                        list(range(self.n_qubits // 2, self.n_qubits)), inplace=True)

        qc = qc.compose(self._conv_layer(self.n_qubits // 4, self.param_vectors['c3']),
                        list(range(self.n_qubits // 4, self.n_qubits // 2)), inplace=True)
        qc = qc.compose(self._pool_layer(self.n_qubits // 4, self.param_vectors['p3']),
                        list(range(self.n_qubits // 4, self.n_qubits // 2)), inplace=True)

        return qc

    def run(self,
            input_data: np.ndarray,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            conv_pool_params: dict,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the variational circuit for a single feature vector.
        Parameters must be supplied as per the classical interface:
          - rotation_params: shape (3 * n_qubits,)
          - entangle_params: shape (n_qubits - 1,)
          - conv_pool_params: mapping from layer name ('c1', 'p1',...) to 1‑D array
        Returns the expectation value of the first‑qubit Z observable.
        """
        # Build parameter mapping
        param_dict = {}
        for i, val in enumerate(rotation_params):
            param_dict[self.rotation_params[i]] = val
        for i, val in enumerate(entangle_params):
            param_dict[self.entangle_params[i]] = val
        for name, arr in conv_pool_params.items():
            vec = self.param_vectors[name]
            for i, val in enumerate(arr):
                param_dict[vec[i]] = val

        # Execute EstimatorQNN
        result = self.qnn.evaluate([input_data], param_dict, shots=shots)
        return np.array(result[0])
