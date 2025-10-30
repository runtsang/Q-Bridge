import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridQuanvolutionNAT(tq.QuantumModule):
    """
    Quantum‑enhanced network that applies a patch‑wise quantum kernel
    followed by a quantum fully‑connected layer and a classical linear head.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4

        # Patch‑wise encoder (ry rotations on each qubit)
        self.patch_encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.patch_random = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.patch_measure = tq.MeasureAll(tq.PauliZ)

        # Quantum fully‑connected layer (from Quantum‑NAT)
        self.q_fc_layer = self.QLayer()

        # Classical head
        self.linear = nn.Linear(4 * 14 * 14, 10)

    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        patches = []

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Prepare classical data for the patch
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # Quantum device for this patch
                qdev = tq.QuantumDevice(
                    self.n_wires, bsz=bsz, device=x.device, record_op=True
                )
                # Encode, random layer, and quantum fully‑connected layer
                self.patch_encoder(qdev, data)
                self.patch_random(qdev)
                self.q_fc_layer(qdev)
                measurement = self.patch_measure(qdev)
                patches.append(measurement.view(bsz, 4))

        patch_features = torch.cat(patches, dim=1)   # shape: (bsz, 4*14*14)
        logits = self.linear(patch_features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQuanvolutionNAT"]
