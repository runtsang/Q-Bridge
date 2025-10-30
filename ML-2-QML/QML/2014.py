"""Quantum implementation of a QCNN with a variational ansatz.

The class builds a full QCNN circuit consisting of parameterised convolution
and pooling layers, wraps it in an EstimatorQNN, and exposes a simple
predict interface.  The ansatz depth is configurable, allowing the circuit
to be scaled up or down for experimentation and direct comparison with
the classical counterpart.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


class QCNNModel:
    """
    Quantum circuit that emulates a QCNN with convolution and pooling layers.
    The circuit is automatically decomposed and wrapped in a
    :class:`qiskit_machine_learning.neural_networks.EstimatorQNN` so it can
    be trained with gradient‑based optimisers.

    The ansatz depth is configurable and each convolution block is
    implemented as a two‑qubit variational unitary.  Pooling blocks reduce
    the qubit count exactly as in the classical counterpart, which allows
    a direct comparison of learned representations.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        conv_depth: int = 3,
        seed: int = 12345,
        device: str = "statevector",
    ) -> None:
        algorithm_globals.random_seed = seed
        self.estimator = Estimator()
        self.num_qubits = num_qubits
        self.conv_depth = conv_depth
        self.circuit = self._build_circuit()
        # Build QNN
        observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    # ------------------------------------------------------------------
    # Variational primitives
    # ------------------------------------------------------------------
    def _conv_unitary(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit variational unitary used inside convolution layers."""
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

    def _pool_unitary(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit unitary used inside pooling layers."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # ------------------------------------------------------------------
    # Layer builders
    # ------------------------------------------------------------------
    def _conv_layer(self, qubits: list[int], param_prefix: str) -> QuantumCircuit:
        """Construct a convolution layer operating on adjacent qubit pairs."""
        qc = QuantumCircuit(len(qubits))
        params = ParameterVector(param_prefix, length=len(qubits) // 2 * 3)
        for i in range(0, len(qubits), 2):
            unitary = self._conv_unitary(
                params[i // 2 * 3 : i // 2 * 3 + 3]
            )
            qc.append(unitary, [qubits[i], qubits[i + 1]])
            qc.barrier()
        return qc

    def _pool_layer(
        self, source_qubits: list[int], sink_qubits: list[int], param_prefix: str
    ) -> QuantumCircuit:
        """Construct a pooling layer that pairs each source with a sink qubit."""
        qc = QuantumCircuit(len(source_qubits) + len(sink_qubits))
        params = ParameterVector(param_prefix, length=len(source_qubits) // 2 * 3)
        for idx, (src, sink) in enumerate(zip(source_qubits, sink_qubits)):
            unitary = self._pool_unitary(
                params[idx * 3 : idx * 3 + 3]
            )
            qc.append(unitary, [src, sink])
            qc.barrier()
        return qc

    # ------------------------------------------------------------------
    # Circuit assembly
    # ------------------------------------------------------------------
    def _build_circuit(self) -> QuantumCircuit:
        """Assemble the full QCNN circuit."""
        # Feature map
        self.feature_map = ZFeatureMap(self.num_qubits)
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(self.feature_map, inplace=True)

        # Ansatz
        self.ansatz = QuantumCircuit(self.num_qubits, name="Ansatz")

        # Build layers according to depth
        qubit_sets = [list(range(self.num_qubits))]
        for d in range(self.conv_depth):
            current = qubit_sets[-1]

            # Convolution
            conv_layer = self._conv_layer(current, f"c{d}")
            self.ansatz.append(conv_layer, current)

            # Pooling
            source = current[: len(current) // 2]
            sink = current[len(current) // 2 :]
            pool_layer = self._pool_layer(source, sink, f"p{d}")
            self.ansatz.append(pool_layer, source + sink)

            # New qubit set for next depth: sink indices
            qubit_sets.append(sink)

        circuit.compose(self.ansatz, inplace=True)
        return circuit

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the QCNN on a batch of classical inputs.

        Parameters
        ----------
        X : np.ndarray of shape (batch, input_dim)
            Classical feature vectors.

        Returns
        -------
        np.ndarray of shape (batch,)
            Probabilities after applying a sigmoid.
        """
        probs = self.qnn.evaluate(inputs=X)
        return np.sigmoid(probs)

__all__ = ["QCNNModel"]
