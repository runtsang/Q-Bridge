"""Quantum QCNN with reusable convolution and pooling blocks.

This module implements a parameter‑sharded QCNN circuit that can be
stretched to arbitrary depth.  The convolution and pooling operations
are defined as small 2‑qubit sub‑circuits and are combined into layers
via the :py:meth:`conv_layer` and :py:meth:`pool_layer` helpers.
The final quantum neural network is exposed through :func:`QCNNEnhanced`
which returns an :class:`~qiskit_machine_learning.neural_networks.EstimatorQNN`.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QCNNEnhancedQModel:
    """Quantum QCNN with sharded parameters and reusable blocks."""

    def __init__(self, qubits: int = 8, num_layers: int = 3) -> None:
        self.qubits = qubits
        self.num_layers = num_layers
        self.estimator = Estimator()
        self.circuit = self._build_circuit()

    # ------------------------------------------------------------------
    #  Basic 2‑qubit sub‑circuits
    # ------------------------------------------------------------------
    def _conv_subcircuit(self, params: ParameterVector) -> QuantumCircuit:
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

    def _pool_subcircuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # ------------------------------------------------------------------
    #  Layer helpers
    # ------------------------------------------------------------------
    def _conv_layer(self, qubits: list[int], layer_idx: int) -> QuantumCircuit:
        """Return a convolution layer acting on the supplied qubits."""
        qc = QuantumCircuit(len(qubits))
        params = ParameterVector(f"c{layer_idx}", length=len(qubits) // 2 * 3)
        idx = 0
        for q1, q2 in zip(qubits[::2], qubits[1::2]):
            qc.append(self._conv_subcircuit(params[idx : idx + 3]), [q1, q2])
            idx += 3
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], layer_idx: int) -> QuantumCircuit:
        """Return a pooling layer that maps sources to sinks."""
        qc = QuantumCircuit(len(sources) + len(sinks))
        params = ParameterVector(f"p{layer_idx}", length=len(sources) * 3)
        idx = 0
        for src, sink in zip(sources, sinks):
            qc.append(self._pool_subcircuit(params[idx : idx + 3]), [src, sink])
            idx += 3
        return qc

    # ------------------------------------------------------------------
    #  Circuit construction
    # ------------------------------------------------------------------
    def _build_circuit(self) -> QuantumCircuit:
        feature_map = ZFeatureMap(self.qubits)
        circuit = QuantumCircuit(self.qubits)
        circuit.compose(feature_map, inplace=True)

        current_qubits = list(range(self.qubits))
        for layer in range(self.num_layers):
            # Convolution
            conv = self._conv_layer(current_qubits, layer)
            circuit.append(conv, current_qubits, inplace=True)

            # Pooling – reduce the number of qubits
            if len(current_qubits) > 2:
                sinks = current_qubits[::2]
                sources = current_qubits[1::2]
                pool = self._pool_layer(sources, sinks, layer)
                circuit.append(pool, sources + sinks, inplace=True)
                current_qubits = sinks

        # Observable for a single‑output regression/classification
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (self.qubits - 1), 1)])
        return circuit

    def get_qnn(self) -> EstimatorQNN:
        """Return a Qiskit EstimatorQNN wrapping the constructed circuit."""
        return EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[],
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )


def QCNNEnhanced(backend: str = "qiskit") -> EstimatorQNN:
    """Return a QCNNEnhanced quantum neural network.

    Parameters
    ----------
    backend : str, optional
        'qiskit' for the quantum model.  The classical backend is not
        available from this module and will raise a ``ValueError``.
    """
    if backend.lower() == "qiskit":
        model = QCNNEnhancedQModel()
        return model.get_qnn()
    raise ValueError(f"Unsupported backend '{backend}'. Only 'qiskit' is supported in the quantum module.")
