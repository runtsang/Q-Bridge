"""Quantum implementation of QCNNGen070.

The class builds a variational quantum circuit that mirrors the
convolution‑pool pattern from the classical QCNNModel and the
quantum ansatz defined in QCNN.py.  It uses a ZFeatureMap for
encoding and an EstimatorQNN to obtain differentiable outputs.

Inspired by:
- The conv_circuit, conv_layer and pool_layer definitions in QCNN.py
- The feature‑map and ansatz construction in QCNN.py
- The build_classifier_circuit function from QuantumClassifierModel.py
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from qiskit import Aer
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QCNNGen070:
    """
    Quantum hybrid QCNN.

    Parameters
    ----------
    num_qubits : int
        Number of qubits representing the input feature map.
    conv_depth : int
        Number of convolution‑pool stages in the ansatz.
    feature_map_depth : int
        Depth of the ZFeatureMap (controls entanglement).
    backend : Backend or None
        Backend to use for the Estimator.  If None, the Aer
        simulator is used.
    shots : int
        Number of shots for measurement.
    threshold : float
        Threshold used in the classical Conv filter; kept for
        compatibility with the Conv.py reference.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        conv_depth: int = 3,
        feature_map_depth: int = 1,
        backend=None,
        shots: int = 1024,
        threshold: float = 0.5,
    ):
        self.num_qubits = num_qubits
        self.conv_depth = conv_depth
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Feature map
        self.feature_map = ZFeatureMap(num_qubits, reps=feature_map_depth)

        # Build ansatz
        self.ansatz = self._build_ansatz()

        # Observable (single‑qubit Z on the first qubit)
        self.observable = SparsePauliOp.from_list(
            [("Z" + "I" * (num_qubits - 1), 1)]
        )

        # Estimator and QNN
        self.estimator = Estimator(backend=self.backend, shots=self.shots)
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    # --------------------------------------------------------------------- #
    #  Quantum ansatz components
    # --------------------------------------------------------------------- #
    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Return a two‑qubit convolution layer circuit."""
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(prefix, length=num_qubits * 3)
        idx = 0
        for q in range(0, num_qubits, 2):
            sub = QuantumCircuit(2)
            sub.rz(-np.pi / 2, 1)
            sub.cx(1, 0)
            sub.rz(param_vec[idx], 0)
            sub.ry(param_vec[idx + 1], 1)
            sub.cx(0, 1)
            sub.ry(param_vec[idx + 2], 1)
            sub.cx(1, 0)
            sub.rz(np.pi / 2, 0)
            qc.append(sub.to_instruction(), [q, q + 1])
            qc.barrier()
            idx += 3
        return qc

    def _pool_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Return a two‑qubit pooling layer circuit."""
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(prefix, length=num_qubits * 3)
        idx = 0
        for q in range(0, num_qubits, 2):
            sub = QuantumCircuit(2)
            sub.rz(-np.pi / 2, 1)
            sub.cx(1, 0)
            sub.rz(param_vec[idx], 0)
            sub.ry(param_vec[idx + 1], 1)
            sub.cx(0, 1)
            sub.ry(param_vec[idx + 2], 1)
            qc.append(sub.to_instruction(), [q, q + 1])
            qc.barrier()
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct the full ansatz with conv and pool layers."""
        qc = QuantumCircuit(self.num_qubits)
        # First convolution
        qc.append(self._conv_layer(self.num_qubits, "c1").to_instruction(), range(self.num_qubits))
        # First pooling
        qc.append(self._pool_layer(self.num_qubits, "p1").to_instruction(), range(self.num_qubits))

        # Second conv on remaining qubits
        remaining = self.num_qubits // 2
        qc.append(self._conv_layer(remaining, "c2").to_instruction(), range(remaining, self.num_qubits))
        # Second pool
        qc.append(self._pool_layer(remaining, "p2").to_instruction(), range(remaining, self.num_qubits))

        # Third conv on last two qubits
        qc.append(self._conv_layer(2, "c3").to_instruction(), range(self.num_qubits - 2, self.num_qubits))
        # Third pool
        qc.append(self._pool_layer(2, "p3").to_instruction(), range(self.num_qubits - 2, self.num_qubits))

        return qc

    # --------------------------------------------------------------------- #
    #  Public interface
    # --------------------------------------------------------------------- #
    def __call__(self, inputs: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Evaluate the quantum neural network.

        Parameters
        ----------
        inputs : array‑like or torch.Tensor
            Shape (batch, num_qubits).  If a torch tensor is
            provided it will be converted to a NumPy array.

        Returns
        -------
        np.ndarray
            Array of shape (batch,) containing the predicted
            probabilities of the target class.
        """
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()
        return self.qnn(inputs).reshape(-1)

    def parameters(self):
        """Return the list of trainable parameters of the ansatz."""
        return self.ansatz.parameters

__all__ = ["QCNNGen070"]
