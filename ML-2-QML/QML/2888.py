"""
Hybrid quantum self‑attention circuit that merges the Qiskit self‑attention block
with a QCNN‑style ansatz.  Parameters are split into rotation/entangle blocks
for the attention part and weight vectors for the convolutional layers.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as EstimatorQ
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridQuantumSelfAttention:
    """
    Quantum hybrid self‑attention block.

    Parameters
    ----------
    n_qubits : int, default 8
        Total qubits – the first half are used for the attention circuit,
        the remainder for the QCNN ansatz.
    """

    def __init__(self, n_qubits: int = 8) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    # ---------- Self‑attention sub‑circuit ----------
    def _attention_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """
        Build a self‑attention style circuit on the first half of the qubits.
        """
        circuit = QuantumCircuit(self.qr, self.cr)
        half = self.n_qubits // 2
        for i in range(half):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        for i in range(half - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure(self.qr[:half], self.cr[:half])
        return circuit

    # ---------- QCNN ansatz helpers ----------
    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit convolution unit from the QCNN seed."""
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
        """Two‑qubit pooling unit from the QCNN seed."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        """Convolution layer that stitches consecutive pairs via the conv circuit."""
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        idx = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.append(self._conv_circuit(params[idx:idx+3]), [q1, q2])
            qc.barrier()
            idx += 3
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        """Pooling layer that reduces the qubit count."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            qc.append(self._pool_circuit(params[idx:idx+3]), [src, snk])
            qc.barrier()
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Compose the full QCNN ansatz on the second half of the qubits."""
        ansatz = QuantumCircuit(self.n_qubits)
        # First convolution + pooling
        ansatz.compose(self._conv_layer(self.n_qubits, "c1"), inplace=True)
        ansatz.compose(self._pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)
        # Second convolution + pooling
        ansatz.compose(self._conv_layer(self.n_qubits//2, "c2"), inplace=True)
        ansatz.compose(self._pool_layer([0,1], [2,3], "p2"), inplace=True)
        return ansatz

    # ---------- Full circuit ----------
    def _full_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
                      ansatz_params: dict[str, np.ndarray]) -> QuantumCircuit:
        """
        Combine the attention block and QCNN ansatz into one circuit.
        """
        circuit = QuantumCircuit(self.qr, self.cr)

        # Attention part
        attention = self._attention_circuit(rotation_params, entangle_params)
        circuit.compose(attention, inplace=True)

        # QCNN ansatz part
        ansatz = self._build_ansatz()
        # Bind ansatz parameters
        for name, vals in ansatz_params.items():
            ansatz = ansatz.bind_parameters({f"{name}_{i}": val for i, val in enumerate(vals)})
        circuit.compose(ansatz, inplace=True)

        return circuit

    def run(self, backend: qiskit.providers.Backend, rotation_params: np.ndarray,
            entangle_params: np.ndarray, ansatz_params: dict[str, np.ndarray],
            shots: int = 1024) -> dict:
        """
        Execute the hybrid circuit and return measurement counts.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            Target backend (simulator or real device).
        rotation_params : np.ndarray
            Parameters for the rotation gates in the attention block.
        entangle_params : np.ndarray
            Parameters for the CRX gates in the attention block.
        ansatz_params : dict
            Mapping from ansatz parameter names to 1‑D arrays of values.
        shots : int, default 1024
            Number of shots for execution.

        Returns
        -------
        dict
            Measurement counts from the backend.
        """
        circuit = self._full_circuit(rotation_params, entangle_params, ansatz_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

def HybridQuantumSelfAttentionFactory() -> HybridQuantumSelfAttention:
    """
    Factory mirroring the original SelfAttention() API but returning the
    hybrid quantum class.
    """
    return HybridQuantumSelfAttention(n_qubits=8)

__all__ = ["HybridQuantumSelfAttention", "HybridQuantumSelfAttentionFactory"]
