"""Quantum QCNN with a variational head.

This module implements :class:`QCNNHybrid`, a quantum‑enhanced
convolutional network.  It mirrors the QCNN QML seed by
combining a Z‑feature map with a variational ansatz composed of
convolutional and pooling‑like blocks.  The network is exposed as a
PyTorch ``nn.Module`` so that it can be trained with standard
optimizers while the forward pass delegates to a
``EstimatorQNN``.

The class exposes a ``forward`` method that accepts a torch tensor
of shape ``(batch, 8)`` and returns a probability tensor of shape
``(batch, 2)``.  A ``predict`` helper returns a NumPy array of
probabilities.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from qiskit import QuantumCircuit, assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QCNNHybrid(nn.Module):
    """Quantum QCNN with variational ansatz and expectation head."""

    def __init__(self, shots: int = 1024) -> None:
        super().__init__()
        self.feature_map = ZFeatureMap(8)
        self.ansatz = self._build_ansatz()
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    # ------------------------------------------------------------------ #
    #  Helper circuits for convolution and pooling layers
    # ------------------------------------------------------------------ #
    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            sub = self._conv_circuit(params[param_index:param_index + 3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        for src, snk in zip(sources, sinks):
            sub = self._pool_circuit(params[param_index:param_index + 3])
            qc.append(sub, [src, snk])
            qc.barrier()
            param_index += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(8)
        # First convolutional layer
        qc.append(self._conv_layer(8, "c1"), range(8))
        # First pooling layer
        qc.append(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8))
        # Second convolutional layer
        qc.append(self._conv_layer(4, "c2"), range(4, 8))
        # Second pooling layer
        qc.append(self._pool_layer([0, 1], [2, 3], "p2"), range(4, 8))
        # Third convolutional layer
        qc.append(self._conv_layer(2, "c3"), range(6, 8))
        # Third pooling layer
        qc.append(self._pool_layer([0], [1], "p3"), range(6, 8))
        return qc

    # ------------------------------------------------------------------ #
    #  Forward pass
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, 8)`` containing raw input
            features that are interpreted as angles for the feature
            map.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, 2)`` with probabilities.
        """
        inputs = x.detach().cpu().numpy()
        exp_vals = self.qnn.predict(inputs)
        probs_pos = torch.sigmoid(torch.tensor(exp_vals, dtype=torch.float32))
        return torch.cat([probs_pos, 1 - probs_pos], dim=-1)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Convenience wrapper that returns numpy probabilities."""
        self.eval()
        with torch.no_grad():
            if not isinstance(inputs, np.ndarray):
                raise TypeError("inputs must be a numpy array")
            exp_vals = self.qnn.predict(inputs)
            probs_pos = torch.sigmoid(torch.tensor(exp_vals, dtype=torch.float32)).numpy()
            return np.column_stack([probs_pos, 1 - probs_pos])

__all__ = ["QCNNHybrid"]
