import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class HybridNatModel(tq.QuantumModule):
    """Quantum‑enhanced Nat model combining quantum encoder, quanvolution, and LSTM."""
    class QEncoder(tq.QuantumModule):
        """Quantum feature encoder with random layers."""
        def __init__(self, n_wires=8):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["8x8_ryzxy"])
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice):
            self.encoder(qdev)
            self.random_layer(qdev)
            return self.measure(qdev)

    class QuanvolutionFilter(tq.QuantumModule):
        """Two‑qubit quantum kernel applied to 2×2 image patches."""
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

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice):
            patches = []
            for r in range(0, 28, 2):
                for c in range(0, 28, 2):
                    data = torch.stack(
                        [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                        dim=1,
                    )
                    self.encoder(qdev, data)
                    self.q_layer(qdev)
                    patches.append(self.measure(qdev).view(x.size(0), 4))
            return torch.cat(patches, dim=1)

    def __init__(self, n_qubits=8, hidden_dim=64):
        super().__init__()
        self.encoder = self.QEncoder(n_wires=n_qubits)
        self.qfilter = self.QuanvolutionFilter()
        self.lstm = nn.LSTM(n_qubits, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.encoder.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        # Quantum encoding of the raw image
        _ = self.encoder(qdev)
        # Quanvolutional feature extraction
        qfeat = self.qfilter(x, qdev)
        # Sequence modeling with classical LSTM
        seq = qfeat.unsqueeze(1)  # (bsz, 1, n_qubits)
        lstm_out, _ = self.lstm(seq)
        out = self.classifier(lstm_out.squeeze(1))
        return self.norm(out)

__all__ = ["HybridNatModel"]
