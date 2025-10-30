"""Quantum self‑attention circuit fused with QCNN layers.

The quantum implementation uses a variational self‑attention block
followed by a stack of convolution and pooling sub‑circuits
adapted from the QCNN example.  The circuit can be executed on
any Qiskit backend and returns measurement counts or expectation
values depending on the estimator used.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from typing import Tuple, List, Union


class QuantumSelfAttentionNet:
    """Variational self‑attention block + QCNN layers."""

    def __init__(self, n_qubits: int = 4, qcnn_qubits: int = 8) -> None:
        self.n_qubits = n_qubits
        self.qcnn_qubits = qcnn_qubits
        self.qr = QuantumRegister(n_qubits + qcnn_qubits, name="q")
        self.cr = ClassicalRegister(n_qubits, name="c")
        self.cqcr = ClassicalRegister(qcnn_qubits, name="cq")

    def _attention_block(
        self,
        rotation_params: ParameterVector,
        entangle_params: ParameterVector,
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(self.qr[:self.n_qubits], self.cr)
        return qc

    def _conv_circuit(self, params: ParameterVector, qubits: List[int]) -> QuantumCircuit:
        """Two‑qubit convolution block from the QCNN reference."""
        qc = QuantumCircuit(self.qr)
        q1, q2 = qubits[0], qubits[1]
        qc.rz(-np.pi / 2, q2)
        qc.cx(q2, q1)
        qc.rz(params[0], q1)
        qc.ry(params[1], q2)
        qc.cx(q1, q2)
        qc.ry(params[2], q2)
        qc.cx(q2, q1)
        qc.rz(np.pi / 2, q1)
        return qc

    def _pool_circuit(self, params: ParameterVector, qubits: List[int]) -> QuantumCircuit:
        """Two‑qubit pooling block from the QCNN reference."""
        qc = QuantumCircuit(self.qr)
        q1, q2 = qubits[0], qubits[1]
        qc.rz(-np.pi / 2, q2)
        qc.cx(q2, q1)
        qc.rz(params[0], q1)
        qc.ry(params[1], q2)
        qc.cx(q1, q2)
        qc.ry(params[2], q2)
        return qc

    def _conv_layer(
        self,
        num_qubits: int,
        params: ParameterVector,
        qubit_offset: int = 0,
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr)
        # Pairwise conv across even‑odd qubits
        for idx in range(0, num_qubits, 2):
            qc.append(
                self._conv_circuit(params[idx * 3 : (idx + 1) * 3], [qubit_offset + idx, qubit_offset + idx + 1]),
                [qubit_offset + idx, qubit_offset + idx + 1],
            )
            qc.barrier()
        # Shifted pairs for overlapping receptive field
        for idx in range(1, num_qubits - 1, 2):
            qc.append(
                self._conv_circuit(params[(idx) * 3 : (idx + 1) * 3], [qubit_offset + idx, qubit_offset + idx + 1]),
                [qubit_offset + idx, qubit_offset + idx + 1],
            )
            qc.barrier()
        return qc

    def _pool_layer(
        self,
        sources: List[int],
        sinks: List[int],
        params: ParameterVector,
        qubit_offset: int = 0,
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr)
        for s, t in zip(sources, sinks):
            qc.append(
                self._pool_circuit(params, [qubit_offset + s, qubit_offset + t]),
                [qubit_offset + s, qubit_offset + t],
            )
            qc.barrier()
        return qc

    def build_ansatz(self, param_dict: dict[str, np.ndarray]) -> QuantumCircuit:
        """Construct the full QCNN‑style ansatz with a preceding self‑attention block."""
        qc = QuantumCircuit(self.qr, self.cr, self.cqcr)

        # Self‑attention block
        rot_params = ParameterVector("α", length=3 * self.n_qubits)
        ent_params = ParameterVector("β", length=self.n_qubits - 1)
        qc.append(self._attention_block(rot_params, ent_params), self.qr[:self.n_qubits])

        # QCNN layers on the remaining qubits
        # Layer 1: conv (8 → 8)
        conv1 = self._conv_layer(self.qcnn_qubits, ParameterVector("γ1", length=3 * self.qcnn_qubits), qubit_offset=self.n_qubits)
        qc.append(conv1, self.qr[self.n_qubits:])

        # Layer 2: pool (8 → 4)
        pool1 = self._pool_layer(
            sources=list(range(0, self.qcnn_qubits, 2)),
            sinks=list(range(1, self.qcnn_qubits, 2)),
            params=ParameterVector("δ1", length=3 * (self.qcnn_qubits // 2)),
            qubit_offset=self.n_qubits,
        )
        qc.append(pool1, self.qr[self.n_qubits:])

        # Layer 3: conv (4 → 4)
        conv2 = self._conv_layer(self.qcnn_qubits // 2, ParameterVector("γ2", length=3 * (self.qcnn_qubits // 2)), qubit_offset=self.n_qubits)
        qc.append(conv2, self.qr[self.n_qubits:])

        # Layer 4: pool (4 → 2)
        pool2 = self._pool_layer(
            sources=list(range(0, self.qcnn_qubits // 2, 2)),
            sinks=list(range(1, self.qcnn_qubits // 2, 2)),
            params=ParameterVector("δ2", length=3 * (self.qcnn_qubits // 4)),
            qubit_offset=self.n_qubits,
        )
        qc.append(pool2, self.qr[self.n_qubits:])

        # Layer 5: conv (2 → 2)
        conv3 = self._conv_layer(self.qcnn_qubits // 4, ParameterVector("γ3", length=3 * (self.qcnn_qubits // 4)), qubit_offset=self.n_qubits)
        qc.append(conv3, self.qr[self.n_qubits:])

        # Layer 6: pool (2 → 1)
        pool3 = self._pool_layer(
            sources=[0],
            sinks=[1],
            params=ParameterVector("δ3", length=3),
            qubit_offset=self.n_qubits,
        )
        qc.append(pool3, self.qr[self.n_qubits:])

        # Measurement of attention qubits
        qc.measure(self.qr[:self.n_qubits], self.cr)
        return qc

    def run(
        self,
        backend: qiskit.providers.Provider,
        param_values: dict[str, np.ndarray],
        shots: int = 1024,
    ) -> dict[str, int]:
        """
        Execute the full circuit on a given backend.

        Parameters
        ----------
        backend : qiskit.providers.Provider
            Execution backend (simulator or device).
        param_values : dict[str, np.ndarray]
            Mapping from parameter names to numpy arrays of values.
        shots : int
            Number of measurement shots.

        Returns
        -------
        dict[str, int]
            Raw measurement counts from the backend.
        """
        circuit = self.build_ansatz(param_values)
        bound_circuit = circuit.bind_parameters(param_values)
        job = qiskit.execute(bound_circuit, backend, shots=shots)
        return job.result().get_counts(bound_circuit)


def SelfAttention() -> QuantumSelfAttentionNet:
    """
    Factory that returns a pre‑configured :class:`QuantumSelfAttentionNet`.
    The default configuration uses 4 qubits for the attention block and
    8 qubits for the QCNN ansatz, matching the classical counterpart.
    """
    return QuantumSelfAttentionNet(n_qubits=4, qcnn_qubits=8)


__all__ = ["SelfAttention", "QuantumSelfAttentionNet"]
