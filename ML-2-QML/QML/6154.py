import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class HybridQuantumNAT(tq.QuantumModule):
    """
    Hybrid quantum model that processes 28×28 images with a patch‑wise quantum convolution,
    then encodes the pooled representation into a 4‑qubit system and applies a
    parameterised quantum circuit.  The design merges the QFCModel encoder and QLayer
    with the QuanvolutionFilter patch‑wise kernel, yielding a 4‑dimensional output.
    """

    class QuantumPatchFilter(tq.QuantumModule):
        """
        Encodes each 2×2 image patch into a 4‑qubit register using a simple
        Ry‑based encoder, applies a random layer, and measures in the Z basis.
        """
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
            self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
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
                    self.random_layer(qdev)
                    measurement = self.measure(qdev)
                    patches.append(measurement.view(bsz, 4))
            return torch.cat(patches, dim=1)  # (bsz, 4*14*14)

    class QLayer(tq.QuantumModule):
        """
        Parameterised quantum circuit that operates on the 4‑qubit system after
        classical encoding.  It is a drop‑in replacement for the QFCModel.QLayer.
        """
        def __init__(self):
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

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.patch_filter = self.QuantumPatchFilter()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Normalised 4‑dimensional output.
        """
        bsz = x.shape[0]
        # Patch‑wise quantum convolution
        patch_features = self.patch_filter(x)   # (bsz, 4*14*14)
        # Global average pooling over patches
        pooled = patch_features.view(bsz, 14 * 14, 4).mean(dim=1)  # (bsz, 4)

        # Encode into qubits
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["HybridQuantumNAT"]
