import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np
import qiskit

class QuanvolutionHybrid(tq.QuantumModule):
    """Quantum implementation of the hybrid model: a 2×2 quantum kernel per image patch
    followed by a fully‑connected quantum layer implemented with a single‑qubit
    parameterized circuit."""
    def __init__(self, n_channels: int = 1, n_classes: int = 10) -> None:
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
        self.classifier = nn.Linear(4 * 14 * 14, n_classes)

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
        flat = torch.cat(patches, dim=1)
        # quantum fully‑connected layer: compute expectation of Z for each feature
        exp = torch.cos(flat)  # shape (bsz, 784)
        logits = self.classifier(exp)
        return F.log_softmax(logits, dim=-1)

    def run_qfc(self, thetas: np.ndarray) -> np.ndarray:
        """Run a quantum fully‑connected layer on a single qubit with external parameters."""
        circuit = qiskit.QuantumCircuit(1)
        circuit.h(0)
        for theta in thetas:
            circuit.ry(theta, 0)
        circuit.measure_all()
        simulator = qiskit.Aer.get_backend('qasm_simulator')
        job = qiskit.execute(circuit, simulator, shots=100)
        result = job.result()
        counts = result.get_counts(circuit)
        probs = np.array(list(counts.values())) / 100
        states = np.array([int(k, 2) for k in counts.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])

__all__ = ["QuanvolutionHybrid"]
