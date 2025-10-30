"""Quantum hybrid model with a tensor‑product encoding, random layer, and dual heads."""
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F

class QFCModel(tq.QuantumModule):
    """
    Unified quantum model echoing the classical counterpart.
    It encodes 4‑wire qubit states, applies a random layer + parameterised gates,
    measures all wires, and maps the resulting amplitudes to two heads.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            # Entangling layer
            for w in range(0, self.n_wires - 1, 2):
                self.crx(qdev, wires=[w, w + 1])

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Dual heads
        self.classifier = nn.Linear(n_wires, 4)
        self.regressor = nn.Linear(n_wires, 1)
        self.norm_cls = nn.BatchNorm1d(4)
        self.norm_reg = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Expects a batch of raw classical states or one‑hot encoded qubit states.
        Returns a dict with classification logits and regression output.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Encode classical input into quantum state
        self.encoder(qdev, x)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return {
            "class_logits": self.norm_cls(self.classifier(features)),
            "regression": self.norm_reg(self.regressor(features).squeeze(-1)),
        }

__all__ = ["QFCModel"]
