"""Quantum hybrid self‑attention module leveraging a QCNN ansatz.

The circuit encodes the input vector via a ZFeatureMap, then applies a
hierarchical QCNN ansatz (convolution + pooling layers).  The expectation
values of Z on each qubit are interpreted as attention weights.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.providers import BackendV2
import math

class HybridSelfAttention:
    """Quantum self‑attention using a QCNN ansatz.

    The ansatz is built from convolutional and pooling sub‑circuits
    inspired by the classical QCNN architecture.  The expectation values
    of the Z observable on each qubit after the ansatz are returned as
    attention weights.  The rotation_params and entangle_params arguments
    are mapped to the ansatz parameters in the order of the convolution
    and pooling layers.
    """
    def __init__(self, n_qubits: int = 8) -> None:
        self.n_qubits = n_qubits
        self.feature_map = ZFeatureMap(n_qubits)
        self._build_ansatz()
        # Observable: Z on every qubit
        self.observable = SparsePauliOp.from_list(
            [("Z" + "I" * (n_qubits - 1 - i), 1) for i in range(n_qubits)]
        )
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    # ---------- sub‑circuits -------------------------------------------------
    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Single 2‑qubit convolutional block."""
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
        """Single 2‑qubit pooling block."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        """Convolutional layer composed of disjoint 2‑qubit blocks."""
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        # First pass: (0,1), (2,3),...
        for i in range(0, num_qubits, 2):
            block = self._conv_circuit(params[i * 3 : i * 3 + 3])
            qc.append(block, [i, i + 1])
        # Second pass: shift by one qubit to overlap
        for i in range(1, num_qubits - 1, 2):
            block = self._conv_circuit(params[i * 3 : i * 3 + 3])
            qc.append(block, [i, i + 1])
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        """Pooling layer that collapses pairs of qubits."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
        for idx, (src, snk) in enumerate(zip(sources, sinks)):
            block = self._pool_circuit(params[idx * 3 : idx * 3 + 3])
            qc.append(block, [src, snk])
        return qc

    def _build_ansatz(self) -> None:
        """Construct the full QCNN ansatz."""
        qc = QuantumCircuit(self.n_qubits)
        # First convolutional layer
        qc.compose(self._conv_layer(self.n_qubits, "c1"), inplace=True)
        # First pooling layer
        qc.compose(
            self._pool_layer(
                list(range(self.n_qubits // 2)),
                list(range(self.n_qubits // 2, self.n_qubits)),
                "p1",
            ),
            inplace=True,
        )
        # Second convolutional layer
        qc.compose(self._conv_layer(self.n_qubits // 2, "c2"), inplace=True)
        # Second pooling layer
        qc.compose(
            self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p2"),
            inplace=True,
        )
        # Third convolutional layer
        qc.compose(self._conv_layer(self.n_qubits // 4, "c3"), inplace=True)
        # Third pooling layer
        qc.compose(self._pool_layer([0], [1], "p3"), inplace=True)
        self.ansatz = qc

    # ---------- public API --------------------------------------------------
    def run(
        self,
        backend: BackendV2,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the QCNN ansatz on the supplied backend and return
        attention weights derived from the expectation values of Z.

        Parameters
        ----------
        backend : qiskit.providers.BackendV2
            Target backend (e.g., AerSimulator).
        rotation_params : np.ndarray
            Parameters for the convolutional blocks.
        entangle_params : np.ndarray
            Parameters for the pooling blocks.
        inputs : np.ndarray
            Classical input vector of shape (n_qubits,).
        shots : int, optional
            Number of shots for the simulation.

        Returns
        -------
        np.ndarray
            Attention weights of shape (n_qubits,).
        """
        # Validate parameter lengths
        total_params = len(self.ansatz.parameters)
        if len(rotation_params) + len(entangle_params)!= total_params:
            raise ValueError(
                f"Expected {total_params} ansatz parameters, "
                f"got {len(rotation_params) + len(entangle_params)}."
            )

        # Bind the input parameters first
        param_dict = dict(zip(self.feature_map.parameters, inputs))
        # Bind the ansatz parameters
        param_dict.update(
            dict(zip(self.ansatz.parameters, np.concatenate([rotation_params, entangle_params])))
        )
        # Build the full circuit
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(self.ansatz, inplace=True)
        circuit.assign_parameters(param_dict, inplace=True)

        # Run the circuit via EstimatorQNN
        result = self.qnn.predict(circuit, backend=backend, shots=shots)
        # result is a list of expectation values; convert to numpy
        expectations = np.array(result).astype(np.float64)
        # Interpret as attention weights
        attn = np.exp(expectations) / np.exp(expectations).sum()
        return attn

def SelfAttention() -> HybridSelfAttention:
    """Factory returning the configured :class:`HybridSelfAttention`."""
    return HybridSelfAttention()
