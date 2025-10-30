from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp

class QuantumHybridClassifier:
    """
    Quantum circuit factory that emulates a quantum convolutional neural network.
    The circuit is built from:
        • A ZFeatureMap encoding of the input data.
        • Convolutional layers implemented with a 2‑qubit template.
        • Pooling layers that down‑sample the qubit register.
        • An ansatz of depth `depth` applying Ry rotations and CZ entanglement.
        • Observables consisting of single‑qubit Z measurements.
    The returned tuple matches the signature used by the classical counterpart:
        (circuit, encoding_params, weight_params, observables)
    """

    @staticmethod
    def build_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        # Feature‑map encoding
        feature_map = ZFeatureMap(num_qubits)

        # 2‑qubit convolution template
        def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

        # Convolution layer over the full register
        def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
            qc = QuantumCircuit(num_qubits)
            qubits = list(range(num_qubits))
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                sub = conv_circuit(params[param_index:param_index+3])
                qc.append(sub, [q1, q2])
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                sub = conv_circuit(params[param_index:param_index+3])
                qc.append(sub, [q1, q2])
                param_index += 3
            return qc

        # 2‑qubit pooling template
        def pool_circuit(params: ParameterVector) -> QuantumCircuit:
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        # Pooling layer over a subset of qubits
        def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
            num_qubits = len(sources) + len(sinks)
            qc = QuantumCircuit(num_qubits)
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for source, sink in zip(sources, sinks):
                sub = pool_circuit(params[param_index:param_index+3])
                qc.append(sub, [source, sink])
                param_index += 3
            return qc

        # Construct the ansatz
        ansatz = QuantumCircuit(num_qubits)
        # First convolution + pooling
        ansatz.compose(conv_layer(num_qubits, "c1"), inplace=True)
        ansatz.compose(pool_layer(list(range(num_qubits)), list(range(num_qubits)), "p1"), inplace=True)

        # Deeper layers
        for i in range(1, depth):
            half = num_qubits // (2 ** i)
            ansatz.compose(conv_layer(half, f"c{i+1}"), inplace=True)
            ansatz.compose(pool_layer(list(range(half)), list(range(half)), f"p{i+1}"), inplace=True)

        # Full circuit
        circuit = QuantumCircuit(num_qubits)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)

        # Observables
        observables = [SparsePauliOp(f"I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

        # Parameter lists
        encoding = list(feature_map.parameters)
        weights = list(ansatz.parameters)

        return circuit, encoding, weights, observables

__all__ = ["QuantumHybridClassifier"]
