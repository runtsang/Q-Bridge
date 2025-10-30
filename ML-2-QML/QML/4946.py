"""Quantum hybrid classifier using QCNN ansatz and feature map."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridClassifier:
    """Quantum classifier that mirrors the classical QCNN architecture.

    The circuit consists of a feature map (ZFeatureMap), a variational ansatz
    built from RealAmplitudes, and a measurement of a Z observable on the
    first qubit.  The class exposes an ``evaluate`` method that returns
    expectation values for a batch of inputs.
    """
    def __init__(self,
                 num_qubits: int,
                 depth: int = 3,
                 observable: SparsePauliOp | None = None) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.observable = observable or SparsePauliOp.from_list(
            [("Z" + "I" * (num_qubits - 1), 1)]
        )
        self._circuit, self.input_params, self.weight_params = self._build_circuit()
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self._circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            interpret=lambda x: x[0],  # single expectation value
            output_shape=1,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector]]:
        """Create the QCNN‑style circuit."""
        feature_map = ZFeatureMap(self.num_qubits)
        ansatz = RealAmplitudes(self.num_qubits, reps=self.depth)

        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)

        input_params = feature_map.parameters
        weight_params = ansatz.parameters
        return circuit, input_params, weight_params

    def evaluate(self, batch: np.ndarray) -> np.ndarray:
        """Return expectation values for a batch of classical data."""
        return np.array(self.qnn.forward(batch))

    def get_parameters(self) -> np.ndarray:
        """Return current variational parameters."""
        return np.array(self.weight_params)

    def set_parameters(self, params: np.ndarray) -> None:
        """Set the variational parameters."""
        param_dict = dict(zip(self.weight_params, params))
        self.qnn.set_parameters(param_dict)

    def predict(self, batch: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions based on expectation value threshold."""
        preds = self.evaluate(batch)
        return (preds >= threshold).astype(np.int64)
