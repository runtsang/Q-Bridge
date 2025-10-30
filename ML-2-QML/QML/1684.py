"""
QCNNModel – a quantum‑classical hybrid QCNN built with Qiskit.

The quantum component is a parameterised ansatz that alternates convolutional
and pooling layers, similar to the original design but with
- depth‑controlled layers,
- a reusable convolutional block,
- a pooling block that measures and discards qubits,
- and a flexible feature‑map.

The class inherits from EstimatorQNN, exposing the same training API
used by Qiskit Machine Learning.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector, Instruction
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QCNNModel(EstimatorQNN):
    """
    Hybrid QCNN model inheriting from EstimatorQNN.

    Parameters
    ----------
    depth : int
        Number of convolution/pooling pairs.  Each pair halves the qubit count.
    num_qubits : int
        Total number of qubits used in the feature map and ansatz.
    feature_dim : int
        Dimensionality of the input data (must equal num_qubits).
    """

    def __init__(self, depth: int = 3, num_qubits: int = 8, feature_dim: int = 8) -> None:
        if feature_dim!= num_qubits:
            raise ValueError("feature_dim must equal num_qubits for a square QCNN.")

        # Create the feature map
        feature_map = ZFeatureMap(num_qubits, reps=1, entanglement="full")
        feature_map_params = feature_map.parameters

        # Build the ansatz
        ansatz = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))

        # Helper to create a convolutional block on a pair of qubits
        def conv_block(q1: int, q2: int, prefix: str) -> Instruction:
            circ = QuantumCircuit(2, name=f"conv_{prefix}")
            p = ParameterVector(prefix, length=3)
            circ.rz(-np.pi / 2, 1)
            circ.cx(1, 0)
            circ.rz(p[0], 0)
            circ.ry(p[1], 1)
            circ.cx(0, 1)
            circ.ry(p[2], 1)
            circ.cx(1, 0)
            circ.rz(np.pi / 2, 0)
            return circ.to_instruction()

        # Helper to create a pooling block on a pair of qubits
        def pool_block(q1: int, q2: int, prefix: str) -> Instruction:
            circ = QuantumCircuit(2, name=f"pool_{prefix}")
            p = ParameterVector(prefix, length=3)
            circ.rz(-np.pi / 2, 1)
            circ.cx(1, 0)
            circ.rz(p[0], 0)
            circ.ry(p[1], 1)
            circ.cx(0, 1)
            circ.ry(p[2], 1)
            return circ.to_instruction()

        # Build layers iteratively
        current_qubits = qubits
        weight_index = 0
        for layer in range(depth):
            # Convolution on adjacent pairs
            conv_instrs = []
            for i in range(0, len(current_qubits), 2):
                q1, q2 = current_qubits[i], current_qubits[i + 1]
                conv_instrs.append(conv_block(q1, q2, f"c{layer}_{i//2}"))
            conv_layer = QuantumCircuit(num_qubits)
            for q1, q2, instr in zip(current_qubits[0::2], current_qubits[1::2], conv_instrs):
                conv_layer.append(instr, [q1, q2])
            ansatz.append(conv_layer, range(num_qubits))

            # Pooling – measure and discard half the qubits
            pool_instrs = []
            for i in range(0, len(current_qubits) - 1, 2):
                q1, q2 = current_qubits[i], current_qubits[i + 1]
                pool_instrs.append(pool_block(q1, q2, f"p{layer}_{i//2}"))
            pool_layer = QuantumCircuit(num_qubits)
            for q1, q2, instr in zip(current_qubits[0::2], current_qubits[1::2], pool_instrs):
                pool_layer.append(instr, [q1, q2])
            ansatz.append(pool_layer, range(num_qubits))

            # Update qubit list – keep first half
            current_qubits = current_qubits[: len(current_qubits) // 2]

        # Final measurement observable on the remaining qubit
        final_qubit = current_qubits[0]
        observable = SparsePauliOp.from_list([(f"Z" + "I" * (num_qubits - 1), 1)])

        super().__init__(
            circuit=ansatz.decompose(),
            observables=observable,
            input_params=feature_map_params,
            weight_params=ansatz.parameters,
            estimator=Estimator(),
        )


def QCNN() -> QCNNModel:
    """Instantiate a default QCNNModel with depth 3."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
