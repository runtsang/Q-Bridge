import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import Statevector
from typing import Sequence

class QuantumKernelMethod:
    """Quantum kernel using a QCNN‑style ansatz built with Qiskit.

    The kernel is evaluated by preparing two feature‑encoded states
    with a ZFeatureMap and then applying a series of convolutional
    and pooling layers inspired by QCNN.  The absolute overlap of
    the resulting statevectors is used as the kernel value.
    """
    def __init__(self):
        self.n_qubits = 8
        self.feature_map = ZFeatureMap(self.n_qubits, reps=1, entanglement='full')
        self.ansatz = self._build_ansatz()

    def _conv_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1, 0)
        qc.rz(0, 0)
        qc.ry(0, 1)
        qc.cx(0, 1)
        qc.ry(0, 1)
        qc.cx(1, 0)
        qc.rz(np.pi/2, 0)
        return qc

    def _pool_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1, 0)
        qc.rz(0, 0)
        qc.ry(0, 1)
        qc.cx(0, 1)
        qc.ry(0, 1)
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # First convolutional and pooling layers
        for i in range(0, self.n_qubits, 2):
            qc.append(self._conv_circuit(), [i, i+1])
        for i in range(1, self.n_qubits-1, 2):
            qc.append(self._conv_circuit(), [i, i+1])
        qc.barrier()
        for src, sink in zip(range(4), range(4, 8)):
            qc.append(self._pool_circuit(), [src, sink])
        qc.barrier()
        # Second convolutional and pooling layers
        for i in range(4, 8, 2):
            qc.append(self._conv_circuit(), [i, i+1])
        for i in range(5, 7, 2):
            qc.append(self._conv_circuit(), [i, i+1])
        qc.barrier()
        for src, sink in zip([4, 5], [6, 7]):
            qc.append(self._pool_circuit(), [src, sink])
        qc.barrier()
        # Third convolutional and pooling layers
        for i in range(6, 8, 2):
            qc.append(self._conv_circuit(), [i, i+1])
        qc.barrier()
        qc.append(self._pool_circuit(), [6, 7])
        return qc

    def _statevector(self, params: np.ndarray) -> Statevector:
        param_map = {p: v for p, v in zip(self.feature_map.parameters, params)}
        full_circuit = self.feature_map.compose(self.ansatz, inplace=False)
        return Statevector.from_instruction(full_circuit, params=param_map)

    def _kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        sv_x = self._statevector(x)
        sv_y = self._statevector(y)
        return abs(np.vdot(sv_x.data, sv_y.data))

    def kernel_matrix(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        """Compute Gram matrix between two datasets."""
        return np.array([[self._kernel(x, y) for y in b] for x in a])

__all__ = ["QuantumKernelMethod"]
