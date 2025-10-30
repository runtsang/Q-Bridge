"""
SelfAttention__gen038.py (QML portion)

This quantum implementation mirrors the classical API while leveraging
QCNN‑style convolution and pooling circuits.  It concludes with an
EstimatorQNN that evaluates the expectation of a Pauli‑Z observable,
providing a differentiable quantum head that can be used in hybrid
learning pipelines.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# --------------------------------------------------------------------------- #
# 1.  QCNN‑style quantum building blocks
# --------------------------------------------------------------------------- #
def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Two‑qubit convolution unit used in the QCNN construction.
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

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Two‑qubit pooling unit used in the QCNN construction.
    """
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """
    Builds a convolutional layer that applies _conv_circuit to each adjacent
    pair of qubits.
    """
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(params[i // 2 * 3 : i // 2 * 3 + 3])
        qc.append(sub, [i, i + 1])
    return qc

def _pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """
    Builds a pooling layer that applies _pool_circuit to each adjacent
    pair of qubits.
    """
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
    for i in range(0, num_qubits, 2):
        sub = _pool_circuit(params[i // 2 * 3 : i // 2 * 3 + 3])
        qc.append(sub, [i, i + 1])
    return qc

# --------------------------------------------------------------------------- #
# 2.  Feature map (ZFeatureMap) and ansatz construction
# --------------------------------------------------------------------------- #
def _build_feature_map(num_qubits: int) -> QuantumCircuit:
    """
    Simple Z‑feature map that maps classical data onto a Pauli‑Z basis.
    """
    from qiskit.circuit.library import ZFeatureMap
    return ZFeatureMap(num_qubits)

def _build_ansatz(num_qubits: int) -> QuantumCircuit:
    """
    QCNN‑style ansatz composed of conv / pool layers.
    """
    qc = QuantumCircuit(num_qubits)
    # Layer 1
    qc.append(_conv_layer(num_qubits, "c1"), range(num_qubits))
    # Layer 2
    qc.append(_pool_layer(num_qubits, "p1"), range(num_qubits))
    # Layer 3 (reduced qubits)
    reduced = num_qubits // 2
    qc.append(_conv_layer(reduced, "c2"), range(reduced))
    qc.append(_pool_layer(reduced, "p2"), range(reduced))
    return qc

# --------------------------------------------------------------------------- #
# 3.  Quantum self‑attention wrapper
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """
    Quantum self‑attention block that accepts rotation and entangle
    parameters, builds a QCNN‑style circuit, and evaluates a Pauli‑Z
    expectation via EstimatorQNN.
    """
    def __init__(self, n_qubits: int = 8, backend=None, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        # Create a reusable estimator
        self.estimator = StatevectorEstimator()
        # Observable for expectation value
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        """
        Assemble the full circuit:
        - Feature map (ZFeatureMap)
        - Ansätze built from QCNN layers
        - Parameterized rotations from rotation_params
        - Entangling operations from entangle_params
        """
        # Feature map
        fm = _build_feature_map(self.n_qubits)
        # Ansatz
        ansatz = _build_ansatz(self.n_qubits)

        circuit = QuantumCircuit(self.n_qubits)
        circuit.append(fm, range(self.n_qubits))
        circuit.append(ansatz, range(self.n_qubits))

        # Apply rotation parameters as single‑qubit rotations
        for i, angle in enumerate(rotation_params):
            circuit.ry(angle, i % self.n_qubits)

        # Entangle parameters as controlled‑RZ gates between neighbours
        for i, angle in enumerate(entangle_params):
            circuit.crz(angle, i % self.n_qubits, (i + 1) % self.n_qubits)

        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int | None = None,
    ) -> np.ndarray:
        """
        Executes the circuit and returns the expectation value of the
        chosen observable.  `shots` can be overridden for a single run.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        compiled = transpile(circuit, self.backend)
        shots = shots or self.shots
        qobj = assemble(compiled, shots=shots)
        result = self.estimator.run(qobj).result
        # Expectation value from the statevector
        ev = result.expectation_value(self.observable).real
        return np.array([ev])

    def __call__(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int | None = None,
    ) -> np.ndarray:
        """
        Allows the instance to be called like a function.
        """
        return self.run(rotation_params, entangle_params, shots)

# --------------------------------------------------------------------------- #
# 4.  Factory
# --------------------------------------------------------------------------- #
def SelfAttention() -> QuantumSelfAttention:
    """
    Factory that returns a pre‑configured QuantumSelfAttention instance.
    """
    return QuantumSelfAttention(n_qubits=8)

__all__ = ["SelfAttention", "QuantumSelfAttention"]
