"""Hybrid quantum estimator that combines a QCNN‑style ansatz with a Qiskit EstimatorQNN.

The :class:`HybridEstimatorQNN` builds a variational circuit inspired by the
QCNN reference, attaches a simple Z‑feature map, and wraps it in a
Qiskit EstimatorQNN object.  It also exposes a quantum kernel routine that
computes the squared overlap between two encoded states, enabling
kernel‑based experiments.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# --------------------------------------------------------------------------- #
# Helper circuits from the QCNN reference
# --------------------------------------------------------------------------- #
def conv_circuit(params):
    """
    Two‑qubit unitary used in each convolutional block.
    """
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

def conv_layer(num_qubits, param_prefix):
    """
    Builds a convolutional layer that acts on adjacent qubit pairs.
    """
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params):
    """
    Two‑qubit pooling operation.
    """
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def pool_layer(sources, sinks, param_prefix):
    """
    Builds a pooling layer that reduces the qubit count.
    """
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

# --------------------------------------------------------------------------- #
# Hybrid quantum estimator
# --------------------------------------------------------------------------- #
class HybridEstimatorQNN:
    """
    Encapsulates a QCNN‑style variational ansatz with a Qiskit EstimatorQNN.
    Provides prediction and quantum kernel evaluation.
    """
    def __init__(self) -> None:
        # Feature map (classical data encoding)
        self.feature_map = ZFeatureMap(8)
        self.feature_map.decompose()  # expand to elementary gates

        # Build the QCNN ansatz
        ansatz = QuantumCircuit(8, name="Ansatz")

        # First convolutional layer
        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)

        # First pooling layer
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

        # Second convolutional layer
        ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

        # Second pooling layer
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

        # Third convolutional layer
        ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

        # Third pooling layer
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(8)
        circuit.compose(self.feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)

        # Observable for regression (simple Z on first qubit)
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        # Quantum estimator
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

        # Store ansatz for kernel computation
        self.ansatz = ansatz
        self.circuit = circuit

    # --------------------------------------------------------------------- #
    # Prediction using the Qiskit EstimatorQNN
    # --------------------------------------------------------------------- #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Forward a NumPy array of samples to the quantum backend.
        """
        return self.estimator_qnn.predict(X)

    # --------------------------------------------------------------------- #
    # Quantum kernel evaluation
    # --------------------------------------------------------------------- #
    def _statevector_from_input(self, x: np.ndarray) -> Statevector:
        """
        Construct the statevector for a single input vector.
        """
        circ = self.circuit.copy()
        # Bind feature‑map parameters
        param_bindings = {p: val for p, val in zip(self.feature_map.parameters, x)}
        # Bind ansatz weight parameters to zero for a pure kernel evaluation
        for w in self.ansatz.parameters:
            param_bindings[w] = 0.0
        circ = circ.bind_parameters(param_bindings)
        return Statevector.from_instruction(circ)

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Quantum kernel: squared overlap between |ψ(x)⟩ and |ψ(y)⟩.
        """
        sv_x = self._statevector_from_input(x)
        sv_y = self._statevector_from_input(y)
        overlap = abs(sv_x.inner(sv_y)) ** 2
        return float(overlap)

# --------------------------------------------------------------------------- #
# Factory function
# --------------------------------------------------------------------------- #
def HybridEstimatorQNN_factory() -> HybridEstimatorQNN:
    """
    Return a ready‑to‑use :class:`HybridEstimatorQNN` instance.
    """
    return HybridEstimatorQNN()

__all__ = ["HybridEstimatorQNN", "HybridEstimatorQNN_factory"]
