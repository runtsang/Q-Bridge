"""Quantum implementation of a hybrid quanvolution network for regression.

This module builds upon the original `QuanvolutionFilter` and
`EstimatorQNN` examples.  The filter operates on 2×2 image patches
using a random two‑qubit quantum kernel.  The resulting feature
vector is fed to a parameterized quantum regression layer
implemented with Qiskit's `EstimatorQNN`.  The overall network is
fully differentiable via the `StatevectorEstimator` backend.
"""

import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


class QuanvolutionHybridNet(tq.QuantumModule):
    """
    Hybrid quantum network: quanvolution filter + EstimatorQNN head.
    """

    def __init__(self, input_channels: int = 1, regression: bool = True):
        super().__init__()
        self.regression = regression
        # Classical pre‑processing: patch extraction using torchquantum
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # EstimatorQNN head
        params = [Parameter("input"), Parameter("weight")]
        self.qc = QuantumCircuit(1)
        self.qc.h(0)
        self.qc.ry(params[0], 0)
        self.qc.rx(params[1], 0)
        observable = SparsePauliOp.from_list([("Y", 1)])
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.qc,
            observables=observable,
            input_params=[params[0]],
            weight_params=[params[1]],
            estimator=self.estimator
        )
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        feature_vector = torch.cat(patches, dim=1)  # shape: (bsz, 784)
        # Use EstimatorQNN on each feature as input, aggregate via sum
        outputs = []
        for i in range(feature_vector.shape[1]):
            inp = feature_vector[:, i]
            out = self.estimator_qnn(inp, self.weight)
            outputs.append(out)
        out = torch.stack(outputs, dim=1).sum(dim=1, keepdim=True)
        if not self.regression:
            out = torch.log_softmax(out, dim=-1)
        return out


__all__ = ["QuanvolutionHybridNet"]
