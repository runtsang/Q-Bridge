import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionFilter(tq.QuantumModule):
    """
    Variational quanvolution filter that processes 2×2 image patches
    with a trainable quantum kernel.  The seed used a RandomLayer
    with fixed parameters; we replace it with a parameterised
    variational circuit (tq.RandomLayer with trainable weights)
    and expose the number of variational layers as a hyper‑parameter.
    """
    def __init__(self,
                 n_wires: int = 4,
                 n_layers: int = 3,
                 var_layer: tq.RandomLayer | None = None):
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps each pixel to a Ry gate
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable variational layer
        self.var_layer = var_layer or tq.RandomLayer(
            n_ops=4 * n_layers, wires=list(range(self.n_wires)), trainable=True
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of images of shape (B, C, H, W) with C=1.

        Returns
        -------
        torch.Tensor
            Concatenated measurement results of shape (B, 4 * 14 * 14).
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
                self.var_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """
    Classifier head that can be purely classical or a small
    variational quantum circuit producing logits.  The original
    seed used a single linear layer; we add an optional quantum
    head that is differentiable via torchquantum.
    """
    def __init__(self,
                 in_features: int,
                 num_classes: int = 10,
                 use_quantum_head: bool = False,
                 n_wires: int = 4,
                 n_layers: int = 2):
        super().__init__()
        self.use_quantum_head = use_quantum_head
        if use_quantum_head:
            # Quantum head that maps the classical feature vector
            # to a set of logits via a variational circuit.
            self.quantum_head = tq.QuantumModule()
            self.quantum_head.n_wires = n_wires
            self.quantum_head.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "ry", "wires": [0]}
                    for i in range(n_wires)
                ]
            )
            self.quantum_head.var_layer = tq.RandomLayer(
                n_ops=2 * n_layers, wires=[0], trainable=True
            )
            self.quantum_head.measure = tq.MeasureAll(tq.PauliZ)
        else:
            self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum_head:
            bsz = x.size(0)
            device = x.device
            qdev = tq.QuantumDevice(self.quantum_head.n_wires,
                                    bsz=bsz, device=device)
            # Truncate or pad features to match n_wires
            feats = x[:, :self.quantum_head.n_wires]
            self.quantum_head.encoder(qdev, feats)
            self.quantum_head.var_layer(qdev)
            logits = self.quantum_head.measure(qdev)
            return F.log_softmax(logits, dim=-1)
        else:
            logits = self.linear(x)
            return F.log_softmax(logits, dim=-1)

class QuanvolutionModule(nn.Module):
    """
    End‑to‑end quantum quanvolution model that mirrors the classical
    counterpart but replaces the filter with a variational circuit and
    optionally uses a quantum classifier head.
    """
    def __init__(self,
                 use_quantum_head: bool = False,
                 n_filter_layers: int = 3,
                 n_classifier_layers: int = 2):
        super().__init__()
        self.filter = QuanvolutionFilter(n_layers=n_filter_layers)
        # 28×28 input → 14×14 patches → 4 values each
        in_features = 4 * 14 * 14
        self.classifier = QuanvolutionClassifier(
            in_features=in_features,
            use_quantum_head=use_quantum_head,
            n_wires=4,
            n_layers=n_classifier_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.filter(x)
        return self.classifier(feats)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier", "QuanvolutionModule"]
