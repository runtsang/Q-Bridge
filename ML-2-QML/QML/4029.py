"""Hybrid quantum‑classical convolutional network."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

class ConvFilter(nn.Module):
    """Classical convolution filter identical to the ML side."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

class HybridQCNN:
    """
    Hybrid quantum‑classical convolutional network.

    If `use_preprocess` is True, a 2×2 classical convolution filter is
    applied first.  The flattened result is then fed into a quantum
    neural network built from convolutional and pooling layers, exactly
    as in the original QCNN but with an optional classical front‑end.
    """
    def __init__(self, use_preprocess: bool = True):
        self.use_preprocess = use_preprocess
        if use_preprocess:
            self.pre = ConvFilter()
        self.qnn = self._build_qnn()

    def _build_qnn(self) -> EstimatorQNN:
        algorithm_globals.random_seed = 12345
        estimator = Estimator()

        def conv_circuit(params):
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
            target = QuantumCircuit(2)
            target.rz(-np.pi / 2, 1)
            target.cx(1, 0)
            target.rz(params[0], 0)
            target.ry(params[1], 1)
            target.cx(0, 1)
            target.ry(params[2], 1)
            return target

        def pool_layer(sources, sinks, param_prefix):
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

        feature_map = ZFeatureMap(8)
        ansatz = QuantumCircuit(8, name="Ansatz")

        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )
        return qnn

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Run the hybrid model on a single data sample.

        Parameters
        ----------
        data : np.ndarray
            Input image of shape (8, 8).

        Returns
        -------
        np.ndarray
            Model prediction as a scalar probability.
        """
        if self.use_preprocess:
            tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            conv_out = self.pre(tensor).detach().numpy().squeeze()
            # Flatten and truncate to 8 features for the 8‑qubit feature map
            features = conv_out.flatten()[:8]
        else:
            features = data.flatten()[:8]
        # EstimatorQNN expects input shape (n_samples, n_features)
        return self.qnn.predict(features.reshape(1, -1))

__all__ = ["HybridQCNN"]
