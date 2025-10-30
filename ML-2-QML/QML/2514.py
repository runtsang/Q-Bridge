import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np

class QuanvolutionFilterQuantum(tq.QuantumModule):
    """
    Quantum patch extractor that applies a random two‑qubit kernel to each
    2×2 image patch.  The implementation follows the original quantum
    filter but is fully differentiable via torchquantum.
    """
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
        """
        ``x`` is a batch of grayscale images of shape (batch, 1, 28, 28).
        The method extracts 2×2 patches, runs them through the quantum
        circuit, and concatenates the results into a single feature vector.
        """
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

class QuantumCircuitWrapper(tq.QuantumModule):
    """
    A single‑qubit parametrised circuit that returns the expectation value
    of Pauli‑Z after applying an Ry rotation with the supplied angle.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 100):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.circuit = tq.QuantumCircuit(n_qubits)
        self.theta = tq.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        ``params`` is a 1‑D tensor of angles (batch,).  The method returns the
        expectation value of Z for each angle.
        """
        batch = params.shape[0]
        device = params.device
        qdev = tq.QuantumDevice(self.n_qubits, bsz=batch, device=device)
        self.circuit.set_parameters({self.theta: params})
        measurement = self.measure(qdev)
        # measurement shape: (batch, n_qubits) with values ±1
        return measurement.mean(dim=1)

class HybridQuantumHead(nn.Module):
    """
    Hybrid head that maps a vector of features to a single probability.
    It first linearly projects the features, then feeds the resulting
    scalar into the quantum circuit to obtain an expectation value,
    which is turned into a probability via a sigmoid.
    """
    def __init__(self,
                 in_features: int,
                 shift: float = np.pi / 2,
                 n_qubits: int = 1,
                 shots: int = 100):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift
        self.circuit = QuantumCircuitWrapper(n_qubits, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x).squeeze()
        exp = self.circuit(logits)
        return torch.sigmoid(exp + self.shift)

__all__ = ["QuanvolutionFilterQuantum", "HybridQuantumHead", "QuantumCircuitWrapper"]
