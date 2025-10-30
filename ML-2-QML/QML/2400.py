import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator

class QuantumNAT:
    """
    Purely quantum QCNN model built with Qiskit.
    Encodes 8‑dimensional input via a Z‑feature map,
    applies a 3‑layer QCNN ansatz (convolution + pooling),
    and returns expectation value of Z on the first qubit.
    """

    def __init__(self, num_qubits: int = 8):
        self.num_qubits = num_qubits
        self.estimator = Estimator()
        self.feature_map = self._build_feature_map()
        self.circuit = self._build_ansatz()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])

    def _build_feature_map(self) -> QuantumCircuit:
        """Z‑feature map used to encode classical data."""
        qreg = QuantumRegister(self.num_qubits)
        qc = QuantumCircuit(qreg)
        for i in range(self.num_qubits):
            qc.rz(Parameter(f"θ_{i}"), qreg[i])
        return qc

    def _conv_circuit(self, params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
        """3‑parameter convolution unit on a pair of qubits."""
        qc = QuantumCircuit(*qubits)
        qc.rz(params[0], qubits[0])
        qc.cx(qubits[1], qubits[0])
        qc.rz(params[1], qubits[0])
        qc.ry(params[2], qubits[1])
        qc.cx(qubits[0], qubits[1])
        return qc

    def _pool_circuit(self, params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
        """3‑parameter pooling unit on a pair of qubits."""
        qc = QuantumCircuit(*qubits)
        qc.rz(params[0], qubits[0])
        qc.cx(qubits[1], qubits[0])
        qc.rz(params[1], qubits[0])
        qc.ry(params[2], qubits[1])
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Convolutional layer composed of pairwise conv circuits."""
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        param_vec = ParameterVector(prefix, length=num_qubits // 2 * 3)
        idx = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            sub = self._conv_circuit(param_vec[idx:idx + 3], [q1, q2])
            qc.append(sub.to_instruction(), [q1, q2])
            idx += 3
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
        """Pooling layer that merges two qubits into one."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(prefix, length=len(sources) * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            sub = self._pool_circuit(param_vec[idx:idx + 3], [src, snk])
            qc.append(sub.to_instruction(), [src, snk])
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Full QCNN ansatz: 3 conv + 3 pool layers."""
        qc = QuantumCircuit(self.num_qubits)
        # First conv layer
        qc.append(self._conv_layer(self.num_qubits, "c1").to_instruction(), range(self.num_qubits))
        # First pool layer
        qc.append(self._pool_layer(list(range(4)), list(range(4, 8)), "p1").to_instruction(), range(self.num_qubits))
        # Second conv layer
        qc.append(self._conv_layer(4, "c2").to_instruction(), range(4, 8))
        # Second pool layer
        qc.append(self._pool_layer([0, 1], [2, 3], "p2").to_instruction(), range(4, 8))
        # Third conv layer
        qc.append(self._conv_layer(2, "c3").to_instruction(), range(6, 8))
        # Third pool layer
        qc.append(self._pool_layer([0], [1], "p3").to_instruction(), range(6, 8))
        return qc

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Evaluate the QCNN on a batch of 8‑dimensional data.
        Parameters
        ----------
        data : np.ndarray
            Shape (batch, 8) with values in [0, 2π].
        Returns
        -------
        np.ndarray
            Expectation values of the observable for each batch element.
        """
        batch = data.shape[0]
        circuits = []
        for i in range(batch):
            qc = QuantumCircuit(self.num_qubits)
            # Feature map
            for j, val in enumerate(data[i]):
                qc.rz(val, j)
            # Append ansatz
            qc.append(self.circuit.to_instruction(), range(self.num_qubits))
            circuits.append(qc)
        # Estimate expectation values
        result = self.estimator.run(circuits, self.observable).result()
        return np.array(result.values)
