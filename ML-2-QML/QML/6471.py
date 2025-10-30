"""QuantumNATEnhanced: quantum variant with variational post‑processing."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """
    Quantum variant of QuantumNATEnhanced.  It encodes a 4‑class
    classification task and a regression task on a 4‑qubit device.
    The encoding uses a 4×4 RyZXY pattern, followed by a
    variational layer and a post‑processing ansatz that maps the
    measurement results to the two tasks.  The module can be trained
    with a weighted combination of cross‑entropy and MSE losses.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            # Trainable single‑qubit rotations
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)
            # Variational post‑processing
            self.var_ry1 = tq.RY(has_params=True, trainable=True)
            self.var_rx2 = tq.RX(has_params=True, trainable=True)
            self.var_cnot = tq.CNOT()
        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)
            # Variational post‑processing
            self.var_ry1(qdev, wires=0)
            self.var_rx2(qdev, wires=1)
            self.var_cnot(qdev, wires=[0, 1])

    def __init__(self, n_wires: int = 4, regression: bool = True):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires + (1 if regression else 0))
        self.regression = regression

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (batch, 1, 28, 28)

        Returns:
            classification logits (batch, 4)
            regression output (batch, 1) if regression=True
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)  # 4x4 encoding input
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)  # shape (batch, n_wires)
        out = self.norm(out)
        if self.regression:
            return out[:, :4], out[:, 4:]
        return out

    def compute_loss(self, logits, targets, loss_fn_cls, loss_fn_reg=None,
                     weight_cls: float = 1.0, weight_reg: float = 1.0):
        """
        Compute weighted loss for classification and regression.
        """
        loss = loss_fn_cls(logits, targets)
        if self.regression and loss_fn_reg is not None:
            loss += weight_reg * loss_fn_reg(logits[:, 4:], targets[:, 4:])
        return loss

__all__ = ["QuantumNATEnhanced"]
