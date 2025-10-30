import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionHybrid(nn.Module):
    """Quantum filter applying a shared variational circuit to each 2x2 patch."""
    def __init__(self, n_wires: int = 4, n_layers: int = 3, n_ops: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.n_ops = n_ops

        # shared Ry encoder for pixel values
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # variational layer with multiple Ry and CNOT gates
        self.var_layer = tq.VariationalLayer(
            n_ops=self.n_ops,
            wires=list(range(self.n_wires)),
            n_layers=self.n_layers,
            param_init="random",
        )

        # measurement of all qubits
        self.measure = tq.MeasureAll(tq.PauliZ)

        # linear head
        self.fc = nn.Linear(self.n_wires * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        patches = []
        x = x.view(bsz, 28, 28)
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
                self.var_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))
        y = torch.cat(patches, dim=1)  # (B, n_wires*14*14)
        logits = self.fc(y)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
