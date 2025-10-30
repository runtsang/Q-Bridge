import torch
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumEncoder(tq.QuantumModule):
    """Quantum variational kernel that applies a random 2‑qubit circuit to each 2×2 patch.

    The encoder mirrors the original QuanvolutionFilter but includes a learnable
    RandomLayer to increase expressivity.  It outputs a 4‑dimensional feature
    vector per patch, yielding a tensor of shape (batch, 4*14*14).
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encode each pixel of a 2×2 patch with an independent Ry gate
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=12, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Reshape to (bsz, 28, 28)
        x = x.view(bsz, 28, 28)

        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Extract 2×2 patch and stack into shape (bsz, 4)
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))

        # Concatenate all patch features into (bsz, 4*14*14)
        return torch.cat(patches, dim=1)

__all__ = ["QuantumEncoder"]
