import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator

class HybridQuantumCircuit:
    """
    Parameterized quantum circuit that combines QCNN‑style convolution and
    pooling layers with a simple feature‑map.  The circuit is built once
    during initialisation and can be executed for a batch of parameters
    using a Qiskit simulator.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.backend = qiskit.Aer.get_backend("statevector_simulator")
        self.circuit = self._build_circuit()
        self.estimator = StatevectorEstimator(backend=self.backend)

    def _conv_layer(self, num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[i*3:(i+1)*3])
            qc.compose(sub, [i, i+1], inplace=True)
        return qc

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

    def _pool_layer(self, num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for i in range(0, num_qubits, 2):
            sub = self._pool_circuit(params[i//2*3:(i//2+1)*3])
            qc.compose(sub, [i, i+1], inplace=True)
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

    def _build_circuit(self):
        # Feature map
        fm = qiskit.circuit.library.ZFeatureMap(self.n_qubits)
        # Ansatz
        ans = QuantumCircuit(self.n_qubits)
        ans.compose(self._conv_layer(self.n_qubits, "c1"), inplace=True)
        ans.compose(self._pool_layer(self.n_qubits, "p1"), inplace=True)
        ans.compose(self._conv_layer(self.n_qubits//2, "c2"), inplace=True)
        ans.compose(self._pool_layer(self.n_qubits//2, "p2"), inplace=True)
        ans.compose(self._conv_layer(self.n_qubits//4, "c3"), inplace=True)
        ans.compose(self._pool_layer(self.n_qubits//4, "p3"), inplace=True)
        # Full circuit
        full = QuantumCircuit(self.n_qubits)
        full.compose(fm, range(self.n_qubits), inplace=True)
        full.compose(ans, range(self.n_qubits), inplace=True)
        return full.decompose()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of parameter vectors.  ``thetas`` is
        expected to be a 1‑D array of length equal to the total number of
        trainable parameters in the circuit.
        """
        if len(thetas)!= self.circuit.num_parameters:
            raise ValueError("Parameter vector length mismatch.")
        param_binds = [{p: v for p, v in zip(self.circuit.parameters, thetas)}]
        result = self.estimator.run(self.circuit, param_binds)[0]
        # Expectation of Z on the first qubit
        exp = result.expectation_value(SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)]))
        return np.array([exp])

__all__ = ["HybridQuantumCircuit"]
