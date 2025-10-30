import numpy as np
import qiskit
from qiskit import execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp

class SelfAttention:
    """
    Quantum self‑attention that embeds the input via a QCNN ansatz and then
    applies a variational attention block.  The circuit is fully parameterized
    by rotation_params and entangle_params and is executed on a Qiskit simulator.
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.feature_map = ZFeatureMap(8)
        self.ansatz = self._build_ansatz()
        self.backend = Aer.get_backend("qasm_simulator")

    # ------------------------------------------------------------------
    # QCNN helpers – identical to the QCNN.py reference
    # ------------------------------------------------------------------
    def _conv_circuit(self, params):
        target = qiskit.QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    def _conv_layer(self, num_qubits, param_prefix):
        qc = qiskit.QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(
                self._conv_circuit(params[param_index : param_index + 3]), [q1, q2]
            )
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(
                self._conv_circuit(params[param_index : param_index + 3]), [q1, q2]
            )
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()
        qc = qiskit.QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    def _pool_circuit(self, params):
        target = qiskit.QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def _pool_layer(self, sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = qiskit.QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(
                self._pool_circuit(params[param_index : param_index + 3]), [source, sink]
            )
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()
        qc = qiskit.QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc

    def _build_ansatz(self):
        # 8‑qubit QCNN ansatz
        ansatz = qiskit.QuantumCircuit(8, name="Ansatz")
        ansatz.compose(self._conv_layer(8, "c1"), range(8), inplace=True)
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
        ansatz.compose(self._conv_layer(4, "c2"), range(4, 8), inplace=True)
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
        ansatz.compose(self._conv_layer(2, "c3"), range(6, 8), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), range(6, 8), inplace=True)
        return ansatz

    # ------------------------------------------------------------------
    # Attention circuit
    # ------------------------------------------------------------------
    def _build_attention_circuit(self, rotation_params, entangle_params):
        qc = qiskit.QuantumCircuit(self.n_qubits)
        # Feature map + ansatz (8 qubits) – we truncate to n_qubits
        fm = self.feature_map
        ans = self.ansatz
        qc.compose(fm, range(fm.num_qubits), inplace=True)
        qc.compose(ans, range(ans.num_qubits), inplace=True)
        if self.n_qubits < fm.num_qubits:
            qc = qc[: self.n_qubits]

        # Rotation layer
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)

        # Entangling layer
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)

        qc.measure_all()
        return qc

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """
        Execute the quantum attention circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Length 3 * n_qubits.
        entangle_params : np.ndarray
            Length n_qubits - 1.
        shots : int, optional
            Number of shots for the simulation.

        Returns
        -------
        dict
            Measurement counts from the simulator.
        """
        qc = self._build_attention_circuit(rotation_params, entangle_params)
        job = execute(qc, self.backend, shots=shots)
        return job.result().get_counts(qc)

__all__ = ["SelfAttention"]
