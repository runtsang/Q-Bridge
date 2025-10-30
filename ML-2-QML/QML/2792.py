"""Hybrid self‑attention + QCNN model – quantum implementation."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# --------------------------------------------------------------------------- #
# Helper: self‑attention style rotation block
# --------------------------------------------------------------------------- #
def _self_attention_block(qr: QuantumRegister, rotation_params: np.ndarray) -> QuantumCircuit:
    """
    Builds a single‑qubit rotation block that mimics the query/key/value
    rotations in the classical self‑attention.

    Parameters
    ----------
    qr : QuantumRegister
        Register containing the qubits to rotate.
    rotation_params : np.ndarray
        Array of shape (3 * n_qubits,) containing Rx, Ry, Rz angles.

    Returns
    -------
    QuantumCircuit
        Circuit containing the rotations.
    """
    qc = QuantumCircuit(qr)
    for i in range(len(qr)):
        idx = 3 * i
        qc.rx(rotation_params[idx], i)
        qc.ry(rotation_params[idx + 1], i)
        qc.rz(rotation_params[idx + 2], i)
    return qc


# --------------------------------------------------------------------------- #
# Helper: entanglement layer (CRX gate pattern)
# --------------------------------------------------------------------------- #
def _entanglement_layer(qr: QuantumRegister, entangle_params: np.ndarray) -> QuantumCircuit:
    """
    Creates an entanglement pattern across adjacent qubits using CRX gates.

    Parameters
    ----------
    qr : QuantumRegister
        Register containing the qubits.
    entangle_params : np.ndarray
        Array of length (n_qubits - 1) with rotation angles.

    Returns
    -------
    QuantumCircuit
        Circuit containing the entanglement gates.
    """
    qc = QuantumCircuit(qr)
    for i in range(len(qr) - 1):
        qc.crx(entangle_params[i], i, i + 1)
    return qc


# --------------------------------------------------------------------------- #
# QCNN‑style convolution layer (two‑qubit unitary)
# --------------------------------------------------------------------------- #
def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Two‑qubit convolution unitary used in QCNN layers.

    Parameters
    ----------
    params : ParameterVector
        Three parameters for the unitary.

    Returns
    -------
    QuantumCircuit
        Two‑qubit circuit implementing the convolution.
    """
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


# --------------------------------------------------------------------------- #
# QCNN‑style pooling layer (two‑qubit unitary)
# --------------------------------------------------------------------------- #
def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Two‑qubit pooling unitary used in QCNN layers.

    Parameters
    ----------
    params : ParameterVector
        Three parameters for the unitary.

    Returns
    -------
    QuantumCircuit
        Two‑qubit circuit implementing the pooling.
    """
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


# --------------------------------------------------------------------------- #
# Construct a convolutional layer for an arbitrary number of qubits
# --------------------------------------------------------------------------- #
def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(
            _conv_circuit(params[param_index : param_index + 3]),
            [q1, q2],
        )
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(
            _conv_circuit(params[param_index : param_index + 3]),
            [q1, q2],
        )
        qc.barrier()
        param_index += 3
    return qc


# --------------------------------------------------------------------------- #
# Construct a pooling layer for an arbitrary number of qubits
# --------------------------------------------------------------------------- #
def _pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, sink in zip(sources, sinks):
        qc.append(
            _pool_circuit(params[param_index : param_index + 3]),
            [src, sink],
        )
        qc.barrier()
        param_index += 3
    return qc


# --------------------------------------------------------------------------- #
# Hybrid quantum circuit that stitches self‑attention and QCNN layers
# --------------------------------------------------------------------------- #
class HybridSelfAttentionQCNN:
    """
    Variational quantum circuit that combines a self‑attention style rotation
    block with a QCNN‑style convolution/pooling stack.  The circuit can be
    executed on a simulator or real backend.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    """

    def __init__(self, n_qubits: int = 8) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

        # Parameter vectors for the different sub‑circuits
        self.rotation_params = ParameterVector("rot", length=3 * n_qubits)
        self.entangle_params = ParameterVector("ent", length=n_qubits - 1)

        # Build the full circuit
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)

        # Self‑attention style block
        qc.append(_self_attention_block(self.qr, self.rotation_params), self.qr)
        qc.append(_entanglement_layer(self.qr, self.entangle_params), self.qr)

        # QCNN convolution/pooling stack (three layers for demonstration)
        conv1 = _conv_layer(self.n_qubits, "c1")
        pool1 = _pool_layer(list(range(self.n_qubits // 2)), list(range(self.n_qubits // 2, self.n_qubits)), "p1")
        conv2 = _conv_layer(self.n_qubits // 2, "c2")
        pool2 = _pool_layer([0, 1], [2, 3], "p2") if self.n_qubits >= 4 else None

        qc.append(conv1, self.qr)
        qc.append(pool1, self.qr)
        qc.append(conv2, self.qr)

        if pool2:
            qc.append(pool2, self.qr)

        # Measurement
        qc.measure(self.qr, self.cr)
        return qc

    def run(
        self,
        backend,
        shots: int = 1024,
        rotation_values: np.ndarray | None = None,
        entangle_values: np.ndarray | None = None,
    ) -> dict:
        """
        Execute the hybrid circuit on the supplied backend.

        Parameters
        ----------
        backend
            Qiskit backend (simulator or real device).
        shots : int, optional
            Number of shots for execution.
        rotation_values : np.ndarray, optional
            Array of shape (3 * n_qubits,) with rotation angles.
        entangle_values : np.ndarray, optional
            Array of shape (n_qubits - 1,) with entanglement angles.

        Returns
        -------
        dict
            Measurement counts.
        """
        # Bind parameters if provided
        param_bindings = {}
        if rotation_values is not None:
            param_bindings.update(
                dict(zip(self.rotation_params, rotation_values.tolist()))
            )
        if entangle_values is not None:
            param_bindings.update(
                dict(zip(self.entangle_params, entangle_values.tolist()))
            )
        bound_circuit = self.circuit.bind_parameters(param_bindings)

        job = qiskit.execute(bound_circuit, backend, shots=shots)
        return job.result().get_counts(bound_circuit)


def HybridSelfAttentionQCNNFactory(n_qubits: int = 8) -> HybridSelfAttentionQCNN:
    """
    Factory that returns a ready‑to‑run hybrid quantum circuit.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits in the circuit (default 8).

    Returns
    -------
    HybridSelfAttentionQCNN
    """
    return HybridSelfAttentionQCNN(n_qubits)


__all__ = ["HybridSelfAttentionQCNN", "HybridSelfAttentionQCNNFactory"]
