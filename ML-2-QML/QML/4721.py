"""Hybrid quantum estimator that composes a feature‑map, an autoencoder circuit, and a QCNN‑style ansatz into a variational QNN."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler

class HybridEstimator:
    """Quantum estimator that bundles a feature map, an autoencoder, and a QCNN ansatz into a single QNN."""
    def __init__(
        self,
        qnn: EstimatorQNN | SamplerQNN,
        observables: Sequence[BaseOperator],
        *,
        shots: Optional[int] = None,
    ) -> None:
        self.qnn = qnn
        self.observables = list(observables)
        self.shots = shots

    @staticmethod
    def autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        """Return a swap‑test autoencoder subcircuit."""
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
        qc.compose(ansatz, list(range(num_latent + num_trash)), inplace=True)
        qc.barrier()
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    @staticmethod
    def conv_circuit(params: Sequence[float], qubits: Sequence[int]) -> QuantumCircuit:
        """One 2‑qubit convolution block used in QCNN."""
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, qubits[1])
        target.cx(qubits[1], qubits[0])
        target.rz(params[0], qubits[0])
        target.ry(params[1], qubits[1])
        target.cx(qubits[0], qubits[1])
        target.ry(params[2], qubits[1])
        target.cx(qubits[1], qubits[0])
        target.rz(np.pi / 2, qubits[0])
        return target

    @staticmethod
    def pool_circuit(params: Sequence[float], qubits: Sequence[int]) -> QuantumCircuit:
        """One 2‑qubit pooling block used in QCNN."""
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, qubits[1])
        target.cx(qubits[1], qubits[0])
        target.rz(params[0], qubits[0])
        target.ry(params[1], qubits[1])
        target.cx(qubits[0], qubits[1])
        target.ry(params[2], qubits[1])
        return target

    @staticmethod
    def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
        """Build a convolutional layer that acts on all adjacent qubit pairs."""
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            sub = HybridEstimator.conv_circuit(params[param_index:param_index + 3], [q1, q2])
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            sub = HybridEstimator.conv_circuit(params[param_index:param_index + 3], [q1, q2])
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    @staticmethod
    def pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
        """Build a pooling layer that acts on source‑sink pairs."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for src, snk in zip(sources, sinks):
            sub = HybridEstimator.pool_circuit(params[param_index:param_index + 3], [src, snk])
            qc.append(sub, [src, snk])
            qc.barrier()
            param_index += 3
        return qc

    @staticmethod
    def build_qcnn_ansatz(num_qubits: int) -> QuantumCircuit:
        """Assemble the full QCNN ansatz with three convolution‑pool pairs."""
        ansatz = QuantumCircuit(num_qubits, name="Ansatz")
        ansatz.compose(HybridEstimator.conv_layer(num_qubits, "c1"), list(range(num_qubits)), inplace=True)
        ansatz.compose(HybridEstimator.pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p1"), list(range(num_qubits)), inplace=True)
        ansatz.compose(HybridEstimator.conv_layer(num_qubits // 2, "c2"), list(range(num_qubits // 2, num_qubits)), inplace=True)
        ansatz.compose(HybridEstimator.pool_layer([0, 1], [2, 3], "p2"), list(range(num_qubits // 2, num_qubits)), inplace=True)
        ansatz.compose(HybridEstimator.conv_layer(num_qubits // 4, "c3"), list(range(num_qubits // 2, num_qubits)), inplace=True)
        ansatz.compose(HybridEstimator.pool_layer([0], [1], "p3"), list(range(num_qubits // 2, num_qubits)), inplace=True)
        return ansatz

    @staticmethod
    def create_qnn(
        num_qubits: int,
        observables: Sequence[BaseOperator],
        *,
        shots: Optional[int] = None,
    ) -> "HybridEstimator":
        """Convenience constructor that builds a feature‑map, QCNN ansatz, and autoencoder, then returns a HybridEstimator instance."""
        feature_map = ZFeatureMap(num_qubits)
        qnn_circuit = QuantumCircuit(num_qubits, name="Full QCNN+AE Circuit")
        qnn_circuit.compose(feature_map, range(num_qubits), inplace=True)
        qnn_circuit.compose(HybridEstimator.build_qcnn_ansatz(num_qubits), range(num_qubits), inplace=True)
        qnn_circuit.compose(HybridEstimator.autoencoder_circuit(3, 2), range(num_qubits), inplace=True)

        if shots is None:
            qnn = EstimatorQNN(
                circuit=qnn_circuit.decompose(),
                observables=observables,
                input_params=feature_map.parameters,
                weight_params=HybridEstimator.build_qcnn_ansatz(num_qubits).parameters,
                estimator=None,
            )
        else:
            sampler = StatevectorSampler()
            qnn = SamplerQNN(
                circuit=qnn_circuit.decompose(),
                input_params=feature_map.parameters,
                weight_params=HybridEstimator.build_qcnn_ansatz(num_qubits).parameters,
                interpret=lambda x: x,
                output_shape=len(observables),
                sampler=sampler,
            )
        return HybridEstimator(qnn, observables, shots=shots)

    def evaluate(self, parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Forward pass through the variational circuit."""
        results: List[List[complex]] = []
        for params in parameter_sets:
            row = self.qnn.forward(params)
            results.append(row)
        return results


__all__ = ["HybridEstimator"]
