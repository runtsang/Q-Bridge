import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator

class HybridFCLQCNN:
    """
    Quantum circuit that implements a QCNN‑style ansatz on 8 qubits.
    After applying a ZFeatureMap and the convolution/pooling layers,
    it returns the expectation values of Pauli‑Z on each qubit.
    """
    def __init__(self,
                 backend: qiskit.providers.Backend | None = None,
                 shots: int = 1024) -> None:
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = self._build_circuit()
        self.estimator = Estimator(backend=self.backend)

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
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            sub = self._conv_circuit(params[param_index:param_index+3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            sub = self._conv_circuit(params[param_index:param_index+3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for src, snk in zip(sources, sinks):
            sub = self._pool_circuit(params[param_index:param_index+3])
            qc.append(sub, [src, snk])
            qc.barrier()
            param_index += 3
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        feature_map = ZFeatureMap(8)
        ansatz = QuantumCircuit(8)

        ansatz.compose(self._conv_layer(8, "c1"), inplace=True)
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
        ansatz.compose(self._conv_layer(4, "c2"), inplace=True)
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), inplace=True)
        ansatz.compose(self._conv_layer(2, "c3"), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)

        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        return circuit

    def run(self, params: np.ndarray) -> np.ndarray:
        """
        Execute the QCNN circuit with the supplied weight parameters.

        Parameters
        ----------
        params : np.ndarray
            Flattened array of all circuit parameters.

        Returns
        -------
        np.ndarray
            Pauli‑Z expectation values for each of the 8 qubits.
        """
        bound = self.circuit.bind_parameters({
            p: val for p, val in zip(self.circuit.parameters, params)
        })
        observables = [SparsePauliOp.from_list([("Z" + "I" * (7-i), 1)]) for i in range(8)]
        result = self.estimator.run(bound, observables=observables)
        return result.expectation.numpy()

def HybridFCLQCNNFactory() -> HybridFCLQCNN:
    """
    Factory that returns the QCNN‑style quantum circuit ready for evaluation.
    """
    return HybridFCLQCNN()

__all__ = ["HybridFCLQCNN", "HybridFCLQCNNFactory"]
