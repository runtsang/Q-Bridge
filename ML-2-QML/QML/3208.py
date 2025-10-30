import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum kernel applied to 2×2 patches of a grayscale image."""
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
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

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
        return torch.cat(patches, dim=1)

class EstimatorQNN(nn.Module):
    """Quantum‑classical hybrid estimator that combines a quanvolution filter
    with a 1‑qubit variational circuit for regression or classification."""
    def __init__(self,
                 use_quanvolution: bool = True,
                 num_classes: int = 10):
        super().__init__()
        self.use_quanvolution = use_quanvolution

        if use_quanvolution:
            self.qfilter = QuanvolutionFilter()
            self.linear = nn.Linear(4 * 14 * 14, 128)
            self.classifier = nn.Linear(128, num_classes)

        # Variational circuit for regression
        self.input_param = Parameter("input")
        self.weight_param = Parameter("weight")
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rx(self.weight_param, 0)

        self.observable = SparsePauliOp.from_list([("Y", 1)])
        self.estimator = StatevectorEstimator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            features = self.qfilter(x)
            hidden = torch.relu(self.linear(features))
            logits = self.classifier(hidden)
            return logits
        else:
            # Map input scalar to circuit parameter
            param_dict = {self.input_param: x.squeeze().item()}
            result = self.estimator.run(circuit=self.circuit,
                                        parameter_values=param_dict)
            expectation = result.result().values()
            return torch.tensor(expectation, device=x.device, dtype=torch.float32)

__all__ = ["QuanvolutionFilter", "EstimatorQNN"]
