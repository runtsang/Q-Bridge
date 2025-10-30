"""
HybridEstimatorQNN – quantum implementation.

Implements a QCNN‑style variational circuit that is wrapped by
`qiskit_machine_learning.neural_networks.EstimatorQNN`.  The circuit
combines a Z‑feature map on 8 qubits with a hierarchical ansatz of
convolution and pooling layers, exactly mirroring the classical
structure in the reference.  The `forward` method accepts a batch of
feature vectors, maps them to the circuit parameters, evaluates the
expectation value of a Z observable, and returns a scalar prediction.

The class is intentionally lightweight; it delegates state‑vector
estimation to Qiskit’s `StatevectorEstimator`, making it suitable
for both local simulators and quantum backends.
"""

from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit import algorithm_globals

class HybridEstimatorQNN:
    """
    Quantum neural network that emulates the QCNN ansatz.

    Parameters
    ----------
    seed : int, default 12345
        Random seed for reproducible parameter initialization.
    """

    def __init__(self, seed: int = 12345) -> None:
        algorithm_globals.random_seed = seed
        self.estimator = StatevectorEstimator()
        self.circuit, self.input_params, self.weight_params = self._build_circuit()

        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self._observable(),
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _observable(self) -> SparsePauliOp:
        """Observable for the regression output (Z on qubit 0)."""
        return SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    def _build_circuit(self):
        """Constructs the full QCNN circuit (feature map + ansatz)."""
        # Feature map: ZFeatureMap on 8 qubits
        feature_map = QuantumCircuit(8)
        for i in range(8):
            feature_map.h(i)
            feature_map.rz(ParameterVector(f"phi_{i}")(), i)

        # Ansatz: hierarchical convolution + pooling layers
        ansatz = QuantumCircuit(8)

        def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=num_qubits * 3)
            for idx, (q1, q2) in enumerate(zip(range(0, num_qubits, 2), range(1, num_qubits, 2))):
                sub = self._conv_circuit(params[3*idx:3*(idx+1)])
                qc.compose(sub, [q1, q2], inplace=True)
            return qc

        def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
            for idx, (src, dst) in enumerate(zip(range(0, num_qubits, 2), range(1, num_qubits, 2))):
                sub = self._pool_circuit(params[3*idx:3*(idx+1)])
                qc.compose(sub, [src, dst], inplace=True)
            return qc

        ansatz.compose(conv_layer(8, "c1"), inplace=True)
        ansatz.compose(pool_layer(8, "p1"), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), inplace=True)
        ansatz.compose(pool_layer(4, "p2"), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), inplace=True)
        ansatz.compose(pool_layer(2, "p3"), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)

        return circuit, feature_map.parameters, ansatz.parameters

    def _conv_circuit(self, params):
        """Convolution unitary on two qubits."""
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

    def _pool_circuit(self, params):
        """Pooling unitary on two qubits."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the quantum circuit on a batch of feature vectors.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch_size, 8), each row maps to the 8 feature map
            parameters `phi_0` … `phi_7`.

        Returns
        -------
        torch.Tensor
            Shape (batch_size, 1), regression predictions.
        """
        batch = inputs.detach().cpu().numpy()
        param_dict = {f"phi_{i}": batch[:, i] for i in range(8)}
        # Run the EstimatorQNN to obtain expectation values
        results = self.qnn.run(param_dict, self.estimator)
        return torch.tensor(results, dtype=torch.float32).unsqueeze(-1)

__all__ = ["HybridEstimatorQNN"]
