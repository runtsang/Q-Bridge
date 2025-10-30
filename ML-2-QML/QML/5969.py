"""Quantum self‑attention + EstimatorQNN hybrid."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

class QuantumSelfAttention:
    """Quantum circuit that implements a self‑attention style block."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        qc = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)

class QuantumAttentionEstimator:
    """
    Hybrid quantum self‑attention circuit feeding into a Qiskit EstimatorQNN.
    The circuit is parameterised by rotation_params and entangle_params; the
    EstimatorQNN learns a mapping from the measurement distribution to a scalar.
    """
    def __init__(self, n_qubits: int = 4, backend=None):
        self.attention = QuantumSelfAttention(n_qubits)
        if backend is None:
            backend = Aer.get_backend("qasm_simulator")
        self.backend = backend

        # Build a minimal EstimatorQNN that will be trained separately.
        params = [Parameter("weight")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        observable = SparsePauliOp.from_list([("Y", 1)])
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[],
            weight_params=params,
            estimator=StatevectorEstimator(),
        )

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        """
        Execute the attention circuit, then evaluate the EstimatorQNN on the
        resulting statevector to obtain a scalar prediction.
        """
        # Build the attention circuit and obtain the statevector
        qc = self.attention._build_circuit(rotation_params, entangle_params)
        state_estimator = StatevectorEstimator()
        state = state_estimator.run(qc).statevector
        # Evaluate EstimatorQNN on the state
        result = self.estimator_qnn.run(state)
        return result

__all__ = ["QuantumAttentionEstimator"]
