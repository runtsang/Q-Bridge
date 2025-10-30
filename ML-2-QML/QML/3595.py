import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionFilter(tq.QuantumModule):
    """
    Quantum implementation of the 2×2 image‑patch filter from the original
    quanvolution example.  It uses a GeneralEncoder followed by a
    RandomLayer and a simultaneous measurement of all qubits.
    """
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
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
        return torch.cat(patches, dim=1)   # shape [B, 784]

class QuantumSelfAttention(tq.QuantumModule):
    """
    Variational quantum circuit that implements a self‑attention style
    operation on a vector of size `n_wires`.  The circuit encodes the
    input vector into rotation angles, applies a trainable entangling
    layer, and measures expectation values of Pauli‑Z.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.var_layer = tq.RandomLayer(n_ops=2 * n_wires, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        self.encoder(qdev, x)
        self.var_layer(qdev)
        return self.measure(qdev)

class QuanvolutionHybrid(tq.QuantumModule):
    """
    Hybrid quantum‑classical model that mirrors the classical version but
    replaces the convolutional backbone with a quantum filter and the
    self‑attention block with a variational quantum circuit.  The
    variational circuit is differentiable via TorchQuantum’s automatic
    differentiation support.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, n_wires: int = 32):
        super().__init__()
        self.n_wires = n_wires
        self.qfilter = QuanvolutionFilter()
        # Reduce 784‑dimensional quantum filter output to `n_wires` for the attention circuit
        self.reduce = nn.Linear(784, n_wires)
        self.attention = QuantumSelfAttention(n_wires=self.n_wires)
        self.classifier = nn.Linear(n_wires, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)          # shape [B, 784]
        reduced = self.reduce(features)     # shape [B, n_wires]
        attn_features = self.attention(reduced)  # shape [B, n_wires]
        logits = self.classifier(attn_features)
        return F.log_softmax(logits, dim=-1)
