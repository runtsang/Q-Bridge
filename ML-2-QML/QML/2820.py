import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

class HybridSelfAttention:
    """
    Quantum self‑attention module that encodes input embeddings as rotation
    angles on individual qubits, entangles them to produce attention
    scores, and uses a tiny quantum neural network (EstimatorQNN) to
    estimate the value vector.  The interface mirrors the classical
    HybridSelfAttention for side‑by‑side experimentation.
    """

    def __init__(self, n_qubits: int, backend=None):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend('qasm_simulator')
        self._estimator_qnn = self._build_estimator_qnn()

    def _build_estimator_qnn(self):
        # Tiny quantum circuit for value estimation
        weight_params = [Parameter(f"w{i}") for i in range(self.n_qubits)]
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.h(i)
            qc.ry(weight_params[i], i)
        observable = SparsePauliOp.from_list([('Y'*self.n_qubits, 1)])
        estimator = StatevectorEstimator()
        return EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[],
            weight_params=weight_params,
            estimator=estimator,
        )

    def _build_attention_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        # Encode embeddings as rotations
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3*i], i)
            qc.ry(rotation_params[3*i+1], i)
            qc.rz(rotation_params[3*i+2], i)
        # Entangle to compute attention
        for i in range(self.n_qubits-1):
            qc.crx(entangle_params[i], i, i+1)
        qc.measure_all()
        return qc

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        """
        Execute the attention circuit and return measurement counts as
        raw attention scores.

        Parameters
        ----------
        rotation_params : np.ndarray
            Flattened rotation angles for each qubit (3 parameters per qubit).
        entangle_params : np.ndarray
            Entanglement rotation angles between adjacent qubits.
        shots : int, optional
            Number of measurement shots.

        Returns
        -------
        dict
            Mapping from measurement bitstring to counts.
        """
        qc = self._build_attention_circuit(rotation_params, entangle_params)
        job = execute(qc, self.backend, shots=shots)
        counts = job.result().get_counts(qc)
        return counts

    def estimate_values(self, weight_params: np.ndarray):
        """
        Use the embedded EstimatorQNN to produce a value vector from
        the given weight parameters.

        Parameters
        ----------
        weight_params : np.ndarray
            Weight parameters for the EstimatorQNN circuit.

        Returns
        -------
        np.ndarray
            Estimated value vector.
        """
        return self._estimator_qnn.predict(weight_params)

__all__ = ["HybridSelfAttention"]
