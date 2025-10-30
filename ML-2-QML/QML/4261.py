"""Quantum version of the combined model."""
import torch
import torchquantum as tq
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Quantum 2x2 patch encoder producing 4‑dimensional feature vector."""
    def __init__(self):
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
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
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
        return torch.cat(patches, dim=1)  # (batch, 4*14*14)

class QuantumSelfAttention(tq.QuantumModule):
    """Parameterised quantum circuit that acts as an attention‑style layer."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.entangle = tq.RandomLayer(n_ops=4, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.entangle(qdev)
        return self.measure(qdev)

class Quanvolution__gen261(tq.QuantumModule):
    """Hybrid quantum‑classical pipeline: quanvolution → quantum attention → qubit‑based regression."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum()
        self.qattn   = QuantumSelfAttention(n_wires=4)
        self.estimator_qnn = self._build_qiskit_estimator()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def _build_qiskit_estimator(self):
        params1 = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params1[0], 0)
        qc.rx(params1[1], 0)
        estimator = StatevectorEstimator()
        observable = SparsePauliOp.from_list([("Y", 1)])
        return QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[params1[0]],
            weight_params=[params1[1]],
            estimator=estimator
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quantum quanvolution
        features = self.qfilter(x)  # (batch, 4*14*14)
        # 2. Reduce to 4‑dimensional vector per sample
        features_mean = features.view(features.size(0), 4, -1).mean(dim=2)  # (batch, 4)
        # 3. Quantum attention on these 4‑dim vectors
        attn_out = self.qattn(features_mean)  # (batch, 4)
        # 4. Collapse to a single scalar per sample (used as input to EstimatorQNN)
        input_vals = attn_out.mean(dim=1)  # (batch,)
        outputs = []
        for val in input_vals:
            params = {
                self.estimator_qnn.input_params[0]: val.item(),
                self.estimator_qnn.weight_params[0]: self.weight.item()
            }
            out = self.estimator_qnn(parameters=params)
            outputs.append(out)
        return torch.tensor(outputs, dtype=torch.float32)

__all__ = ["Quanvolution__gen261"]
