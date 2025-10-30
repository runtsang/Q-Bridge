import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum quanvolution filter applying a convolution followed by a pooling layer to each 2×2 image patch."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable parameters for the convolution and pooling stages
        self.conv_params = nn.Parameter(torch.randn(3))
        self.pool_params = nn.Parameter(torch.randn(3))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def _apply_conv(self, qdev: tq.QuantumDevice) -> None:
        qdev.rz(-np.pi / 2, 1)
        qdev.cx(1, 0)
        qdev.rz(self.conv_params[0], 0)
        qdev.ry(self.conv_params[1], 1)
        qdev.cx(0, 1)
        qdev.ry(self.conv_params[2], 1)
        qdev.cx(1, 0)
        qdev.rz(np.pi / 2, 0)

    def _apply_pool(self, qdev: tq.QuantumDevice) -> None:
        qdev.rz(-np.pi / 2, 1)
        qdev.cx(1, 0)
        qdev.rz(self.pool_params[0], 0)
        qdev.ry(self.pool_params[1], 1)
        qdev.cx(0, 1)
        qdev.ry(self.pool_params[2], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self._apply_conv(qdev)
                self._apply_pool(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

def QCNN() -> EstimatorQNN:
    """Builds a QCNN ansatz comprising convolution and pooling layers."""
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    def conv_layer(params: ParameterVector) -> QuantumCircuit:
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

    def pool_layer(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def conv_block(num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolution")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(conv_layer(params[param_index : param_index + 3]), [q1, q2], inplace=True)
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.compose(conv_layer(params[param_index : param_index + 3]), [q1, q2], inplace=True)
            param_index += 3
        return qc

    def pool_block(num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Pooling")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        for i in range(0, num_qubits, 2):
            qc.compose(pool_layer(params[param_index : param_index + 3]), [qubits[i], qubits[i + 1]], inplace=True)
            param_index += 3
        return qc

    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(conv_block(8, "c1"), inplace=True)
    ansatz.compose(pool_block(8, "p1"), inplace=True)
    ansatz.compose(conv_block(4, "c2"), inplace=True)
    ansatz.compose(pool_block(4, "p2"), inplace=True)
    ansatz.compose(conv_block(2, "c3"), inplace=True)
    ansatz.compose(pool_block(2, "p3"), inplace=True)

    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QuantumQuanvolutionFilter", "QCNN"]
