"""Quantum Self‑Attention with QCNN‑style ansatz.

The quantum implementation builds a parameterised QCNN circuit
(conv + pool layers) followed by a feature map.  The attention
scores are extracted by measuring a Pauli‑Z observable on a
subset of qubits, emulating the classical dot‑product attention.
The API mirrors the classical `run` method and can be executed
on any `qiskit` backend.

Typical usage::

    from SelfAttention__gen023 import SelfAttention
    q_attn = SelfAttention()
    result = q_attn.run(
        backend=Aer.get_backend("qasm_simulator"),
        rotation_params=np.random.rand(12),
        entangle_params=np.random.rand(3),
        shots=1024
    )

"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


class QuantumSelfAttentionQCNN:
    """
    Quantum self‑attention module that emulates a QCNN ansatz and
    extracts attention scores via a measurement of a weighted
    Pauli‑Z observable.

    Parameters
    ----------
    n_qubits : int, default 8
        Number of qubits used for the QCNN ansatz.
    """

    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

        # Build the QCNN ansatz
        self.ansatz = self._build_qcnn_ansatz()
        self.feature_map = self._build_feature_map()
        self.combined_circuit = self._compose_circuit()

        # Estimator for variational evaluation
        self.estimator = Estimator()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])

    def _build_conv_layer(self, params: ParameterVector) -> QuantumCircuit:
        """Single convolution block used in the QCNN."""
        qc = QuantumCircuit(self.n_qubits)
        for i in range(0, self.n_qubits, 2):
            qc.cx(i, i + 1)
            qc.rz(params[3 * i], i)
            qc.ry(params[3 * i + 1], i + 1)
            qc.cx(i + 1, i)
        return qc

    def _build_pool_layer(self, params: ParameterVector) -> QuantumCircuit:
        """Single pooling block used in the QCNN."""
        qc = QuantumCircuit(self.n_qubits)
        for i in range(0, self.n_qubits, 2):
            qc.cx(i, i + 1)
            qc.rz(params[3 * i], i)
            qc.ry(params[3 * i + 1], i + 1)
        return qc

    def _build_qcnn_ansatz(self) -> QuantumCircuit:
        """Constructs a 3‑layer QCNN ansatz."""
        qc = QuantumCircuit(self.n_qubits)
        param_prefix = "θ"
        params = ParameterVector(param_prefix, length=self.n_qubits * 3 * 3)

        # Three conv layers
        for layer in range(3):
            qc.append(self._build_conv_layer(params[layer * self.n_qubits * 3: (layer + 1) * self.n_qubits * 3]), range(self.n_qubits))
            qc.barrier()

        # Three pool layers
        for layer in range(3):
            qc.append(self._build_pool_layer(params[(layer + 3) * self.n_qubits * 3: (layer + 4) * self.n_qubits * 3]), range(self.n_qubits))
            qc.barrier()

        return qc

    def _build_feature_map(self) -> QuantumCircuit:
        """Simple Z‑feature map for data encoding."""
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.ry(0.1, i)  # placeholder encoding
        return qc

    def _compose_circuit(self) -> QuantumCircuit:
        """Combines feature map and QCNN ansatz."""
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(self.ansatz, inplace=True)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        """
        Execute the quantum attention circuit.

        Parameters
        ----------
        backend : qiskit backend
            Execution backend (e.g., Aer.get_backend('qasm_simulator')).
        rotation_params : np.ndarray
            Parameters for the rotation gates in the QCNN ansatz.
        entangle_params : np.ndarray
            Parameters for the entangling gates (unused in this simple
            implementation but kept for API compatibility).
        shots : int, default 1024
            Number of measurement shots.

        Returns
        -------
        dict
            Measurement counts of the final qubits, which can be
            interpreted as attention scores.
        """
        # Bind parameters
        param_bindings = {}
        idx = 0
        for sym in self.combined_circuit.parameters:
            param_bindings[sym] = rotation_params[idx]
            idx += 1

        bound_circuit = self.combined_circuit.bind_parameters(param_bindings)
        job = qiskit.execute(bound_circuit, backend, shots=shots)
        return job.result().get_counts(bound_circuit)


def SelfAttention() -> QuantumSelfAttentionQCNN:
    """
    Factory function mirroring the classical counterpart.
    Instantiates a `QuantumSelfAttentionQCNN` with a default qubit count.
    """
    return QuantumSelfAttentionQCNN(n_qubits=8)

__all__ = ["SelfAttention"]
