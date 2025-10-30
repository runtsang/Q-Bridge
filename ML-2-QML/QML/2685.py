"""Hybrid quantum self‑attention + QCNN circuit.

The quantum model first encodes classical data with a ZFeatureMap,
then applies a self‑attention style block of single‑qubit rotations
and controlled‑X gates, followed by the convolution and pooling
layers of a QCNN.  The circuit is wrapped in an EstimatorQNN so
it can be trained with gradient‑based optimizers.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class HybridQuantumSelfAttentionQCNN:
    """
    Quantum circuit that merges a self‑attention pattern with a QCNN
    ansatz.  It is designed to be compatible with the classical
    HybridSelfAttentionQCNNModel for end‑to‑end hybrid training.
    """

    def __init__(self, n_qubits: int = 8, backend=None):
        """
        Parameters
        ----------
        n_qubits : int, optional
            Number of qubits used in the circuit.
        backend : optional
            Backend for execution; defaults to Aer qasm_simulator.
        """
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

        # Parameter vectors
        self.att_params = ParameterVector("θ_att", length=3 * n_qubits)
        self.conv_params = ParameterVector("θ_conv", length=n_qubits * 3)
        self.pool_params = ParameterVector("θ_pool", length=(n_qubits // 2) * 3)

    # ------------------------------------------------------------------
    # Self‑attention sub‑circuit
    # ------------------------------------------------------------------
    def _attention_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """
        Builds a self‑attention style circuit with rotations and
        controlled‑X gates across adjacent qubits.
        """
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(params[3 * i], i)
            qc.ry(params[3 * i + 1], i)
            qc.rz(params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(params[self.n_qubits + i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    # ------------------------------------------------------------------
    # QCNN convolution layer
    # ------------------------------------------------------------------
    def _conv_layer(self, num_qubits: int, params: ParameterVector) -> QuantumCircuit:
        """
        Applies a convolutional block to pairs of qubits.
        """
        qc = QuantumCircuit(num_qubits)
        for i in range(0, num_qubits, 2):
            qc.cx(i, i + 1)
            qc.rz(params[3 * (i // 2)], i)
            qc.ry(params[3 * (i // 2) + 1], i + 1)
            qc.cx(i + 1, i)
            qc.ry(params[3 * (i // 2) + 2], i + 1)
            qc.cx(i, i + 1)
        return qc

    # ------------------------------------------------------------------
    # QCNN pooling layer
    # ------------------------------------------------------------------
    def _pool_layer(self, num_qubits: int, params: ParameterVector) -> QuantumCircuit:
        """
        Applies a pooling block that entangles and discards half of the qubits.
        """
        qc = QuantumCircuit(num_qubits)
        for i in range(0, num_qubits, 2):
            qc.cx(i, i + 1)
            qc.rz(params[3 * (i // 2)], i)
            qc.ry(params[3 * (i // 2) + 1], i + 1)
            qc.cx(i + 1, i)
            qc.ry(params[3 * (i // 2) + 2], i + 1)
        return qc

    # ------------------------------------------------------------------
    # Full ansatz construction
    # ------------------------------------------------------------------
    def build_ansatz(self) -> QuantumCircuit:
        """
        Builds the complete circuit: feature map → attention → conv/pool
        layers.  The circuit is returned in a form suitable for use with
        EstimatorQNN.
        """
        # Feature map for data encoding
        feature_map = ZFeatureMap(self.n_qubits)
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(feature_map, inplace=True)

        # Self‑attention block
        circuit.compose(self._attention_circuit(self.att_params), inplace=True)

        # Convolution–pooling stages
        conv1 = self._conv_layer(self.n_qubits, self.conv_params[0:self.n_qubits * 3])
        pool1 = self._pool_layer(self.n_qubits // 2, self.pool_params[0:(self.n_qubits // 2) * 3])

        circuit.compose(conv1, inplace=True)
        circuit.compose(pool1, inplace=True)

        # Additional layers can be appended here if desired
        return circuit

    # ------------------------------------------------------------------
    # EstimatorQNN wrapper
    # ------------------------------------------------------------------
    def get_qnn(self) -> EstimatorQNN:
        """
        Returns an EstimatorQNN instance that can be trained with
        gradient‑based optimizers.
        """
        estimator = Estimator()
        circuit = self.build_ansatz()
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)])
        qnn = EstimatorQNN(
            circuit=circuit,
            observables=observable,
            input_params=ZFeatureMap(self.n_qubits).parameters,
            weight_params=self.att_params + self.conv_params + self.pool_params,
            estimator=estimator,
        )
        return qnn


def HybridQuantumSelfAttentionQCNN() -> HybridQuantumSelfAttentionQCNN:
    """
    Factory that returns a ready‑to‑use instance of the hybrid quantum model.
    """
    return HybridQuantumSelfAttentionQCNN()
