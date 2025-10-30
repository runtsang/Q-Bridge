from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from dataclasses import dataclass

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class FraudDetectionHybridQuantum:
    """
    Quantum counterpart to the classical FraudDetectionHybrid.
    Builds a QCNN‑style variational circuit that accepts image data via
    a ZFeatureMap and uses convolution‑ and pooling‑like blocks.
    Photonic parameters from a FraudLayerParameters instance are
    mapped to specific rotation angles in the ansatz.
    """
    def __init__(self, params: FraudLayerParameters) -> None:
        self.params = params
        self.circuit = self._build_circuit()
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=[SparsePauliOp.from_list([("Z", 1)])],
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )
        self._map_photonic_params()

    def _conv_block(self, qubits: list[int], prefix: str) -> QuantumCircuit:
        """Two‑qubit convolution block with 3 rotation gates."""
        qc = QuantumCircuit(len(qubits))
        params = ParameterVector(prefix, length=len(qubits) * 3)
        for i, q in enumerate(qubits):
            qc.rz(params[i * 3], q)
            qc.ry(params[i * 3 + 1], q)
            qc.cx(q, (q + 1) % len(qubits))
            qc.ry(params[i * 3 + 2], q)
        return qc

    def _pool_block(self, qubits: list[int], prefix: str) -> QuantumCircuit:
        """Pooling block that entangles adjacent qubits."""
        qc = QuantumCircuit(len(qubits))
        params = ParameterVector(prefix, length=len(qubits) // 2 * 3)
        idx = 0
        for i in range(0, len(qubits), 2):
            qc.cx(qubits[i], qubits[i + 1])
            qc.rz(params[idx], qubits[i])
            qc.ry(params[idx + 1], qubits[i + 1])
            qc.cz(qubits[i], qubits[i + 1])
            idx += 3
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        """Assembles the QCNN ansatz with a ZFeatureMap."""
        n_qubits = 8
        self.feature_map = ZFeatureMap(n_qubits)
        self.ansatz = QuantumCircuit(n_qubits)

        # First convolution layer
        self.ansatz.compose(self._conv_block(list(range(n_qubits)), "c1"), inplace=True)

        # First pooling layer
        self.ansatz.compose(self._pool_block(list(range(n_qubits)), "p1"), inplace=True)

        # Second convolution on reduced qubits
        reduced = [i for i in range(n_qubits) if i % 2 == 0]
        self.ansatz.compose(self._conv_block(reduced, "c2"), inplace=True)

        # Second pooling
        self.ansatz.compose(self._pool_block(reduced, "p2"), inplace=True)

        # Third convolution on remaining qubit
        self.ansatz.compose(self._conv_block([reduced[0]], "c3"), inplace=True)

        # Full circuit with feature map
        circuit = QuantumCircuit(n_qubits)
        circuit.compose(self.feature_map, range(n_qubits), inplace=True)
        circuit.compose(self.ansatz, range(n_qubits), inplace=True)
        return circuit

    def _map_photonic_params(self) -> None:
        """
        Map photonic parameters to the ansatz rotation angles.
        This is a heuristic mapping; a real implementation would
        require domain‑specific knowledge.
        """
        mapping = {
            "c1_0": self.params.bs_theta,
            "c1_1": self.params.bs_phi,
            "c2_0": self.params.phases[0],
            "c2_1": self.params.phases[1],
            "p1_0": self.params.squeeze_r[0],
            "p1_1": self.params.squeeze_r[1],
            "c3_0": self.params.displacement_r[0],
            "c3_1": self.params.displacement_r[1],
        }
        for param in self.ansatz.parameters:
            if param.name in mapping:
                param.assign(mapping[param.name])

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Evaluate the quantum neural network on input data.

        Parameters
        ----------
        data
            Array of shape (batch, 1, 28, 28) matching the classical
            feature extractor.

        Returns
        -------
        np.ndarray
            Array of predicted fraud scores.
        """
        return self.qnn.predict(data)

__all__ = ["FraudDetectionHybridQuantum", "FraudLayerParameters"]
