from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit import Parameter

class HybridSelfAttentionQuantum:
    """
    Quantum‑classical hybrid self‑attention that mirrors the classical
    HybridSelfAttention using Qiskit circuits for each sub‑module.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")

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

    def _pool_circuit(self, params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits, params):
        qc = QuantumCircuit(num_qubits)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = self._conv_circuit(params[idx : idx + 3])
            qc.append(sub, [q1, q2])
            idx += 3
        return qc

    def _pool_layer(self, num_qubits, params):
        qc = QuantumCircuit(num_qubits)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = self._pool_circuit(params[idx : idx + 3])
            qc.append(sub, [q1, q2])
            idx += 3
        return qc

    def _fcl_circuit(self, theta):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(theta, 0)
        qc.measure_all()
        return qc

    def _estimator_qnn(self, theta):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(theta, 0)
        qc.rx(theta, 0)
        qc.measure_all()
        return qc

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        conv_params: np.ndarray,
        pool_params: np.ndarray,
        fcl_theta: float,
        est_theta: float,
        shots: int = 1024,
    ):
        """
        Build and execute a quantum circuit that combines self‑attention,
        a QCNN layer, an FCL sub‑circuit and an EstimatorQNN sub‑circuit.
        """
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Self‑attention rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entanglement
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        # QCNN convolution
        conv_layer = self._conv_layer(self.n_qubits, conv_params)
        circuit.append(conv_layer, qr)

        # QCNN pooling
        pool_layer = self._pool_layer(self.n_qubits, pool_params)
        circuit.append(pool_layer, qr)

        # FCL sub‑circuit on the first qubit
        fcl_circ = self._fcl_circuit(fcl_theta)
        circuit.append(fcl_circ, [qr[0]])

        # EstimatorQNN sub‑circuit on the first qubit
        est_circ = self._estimator_qnn(est_theta)
        circuit.append(est_circ, [qr[0]])

        circuit.measure(qr, cr)

        job = execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

__all__ = ["HybridSelfAttentionQuantum"]
