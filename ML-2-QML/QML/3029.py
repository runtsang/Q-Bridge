from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np
import qiskit
from typing import Iterable

class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Quantum patch extractor applying a random 2‑qubit kernel to each 2×2 image patch."""
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

class QuantumFullyConnectedLayer:
    """Single‑qubit parameterised circuit whose expectation value implements a scalar mapping."""
    def __init__(self, backend=None, shots=100):
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._circuit = qiskit.QuantumCircuit(1)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(0)
        self._circuit.ry(self.theta, 0)
        self._circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

class QuanvolutionFCLQuantum(nn.Module):
    """Hybrid quantum‑classical classifier combining a quantum filter with a quantum fully‑connected layer."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum()
        self.qfc = QuantumFullyConnectedLayer()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)  # shape (batch, 4*14*14)
        # Apply quantum fully connected layer element‑wise
        q_values = []
        for val in features.view(-1):
            q_values.append(self.qfc.run([val.item()])[0])
        q_values = torch.tensor(q_values, dtype=torch.float32, device=x.device).view(features.size())
        logits = self.linear(q_values)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilterQuantum", "QuantumFullyConnectedLayer", "QuanvolutionFCLQuantum"]
