"""Hybrid quantum‑classical model that fuses a torchquantum quanvolution filter
with a qiskit EstimatorQNN variational circuit. The filter extracts 2×2 patches,
their mean is used as the input to a single‑qubit variational circuit, and
the expectation value of a Pauli‑Y observable is returned as the output.
"""

import torch
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
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


class QuantumEstimator:
    """EstimatorQNN using a single‑qubit variational circuit."""
    def __init__(self) -> None:
        # Define a simple 1‑qubit circuit with an input rotation and a trainable weight
        self.input_param = Parameter("input")
        self.weight_param = Parameter("weight")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(self.input_param, 0)
        qc.rx(self.weight_param, 0)
        observable = SparsePauliOp.from_list([("Y", 1)])

        # Use state‑vector estimator for deterministic outputs
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[self.input_param],
            weight_params=[self.weight_param],
            estimator=self.estimator,
        )

    def __call__(self, inputs: list[float]) -> torch.Tensor:
        # `inputs` is a list of mean feature values for each batch sample
        output = self.estimator_qnn(inputs)  # returns a numpy array
        return torch.tensor(output, dtype=torch.float32)


class QuanvolutionHybridQuantum:
    """Hybrid quantum‑classical model combining the quanvolution filter
    with a quantum estimator head."""
    def __init__(self) -> None:
        self.filter = QuanvolutionFilterQuantum()
        self.estimator = QuantumEstimator()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Extract quantum patch features
        features = self.filter(x)  # shape: (batch, 4*14*14)
        # For demonstration, use the mean of all features as the input to the estimator
        mean_feature = features.mean(dim=1).tolist()
        # Pass the mean through the quantum estimator
        return self.estimator(mean_feature)


__all__ = ["QuanvolutionFilterQuantum", "QuantumEstimator", "QuanvolutionHybridQuantum"]
