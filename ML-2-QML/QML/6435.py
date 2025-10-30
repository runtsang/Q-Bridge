import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import qiskit
import numpy as np
from typing import Optional

class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
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

class QuantumFullyConnectedLayer(tq.QuantumModule):
    """Parameterised quantum circuit that maps an input vector to a set of output expectation values."""
    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 shots: int = 100,
                 backend: Optional[qiskit.providers.Provider] = None):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.weights = nn.Parameter(torch.randn(n_outputs, n_inputs))

    def _expectation_from_circuit(self, theta: float) -> float:
        circ = qiskit.QuantumCircuit(1)
        circ.h(0)
        circ.ry(theta, 0)
        circ.measure_all()
        job = qiskit.execute(circ, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(circ)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        return float(np.sum(states * probs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        logits = torch.empty(batch_size, self.n_outputs, device=x.device)
        for i in range(self.n_outputs):
            theta = torch.matmul(x, self.weights[i])  # shape (batch,)
            expectations = []
            for theta_val in theta.detach().cpu().numpy():
                expectations.append(self._expectation_from_circuit(theta_val))
            logits[:, i] = torch.tensor(expectations, device=x.device)
        return logits

class QuanvolutionHybrid(tq.QuantumModule):
    """Hybrid quantum neural network that mirrors the classical QuanvolutionHybrid architecture but replaces the linear head with a quantum fully‑connected layer."""
    def __init__(self,
                 n_classes: int = 10,
                 hidden_dim: int = 128,
                 shots: int = 100):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.fc = nn.Linear(4 * 14 * 14, hidden_dim)
        self.qfc = QuantumFullyConnectedLayer(hidden_dim, n_classes,
                                              shots=shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        hidden = self.fc(features)
        logits = self.qfc(hidden)
        return F.log_softmax(logits, dim=-1)

# Backwards compatibility with the original module name
QuanvolutionClassifier = QuanvolutionHybrid

__all__ = ["QuanvolutionFilter", "QuantumFullyConnectedLayer", "QuanvolutionHybrid", "QuanvolutionClassifier"]
