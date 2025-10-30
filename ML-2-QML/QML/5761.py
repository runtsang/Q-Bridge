import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

class SelfAttention(tq.QuantumModule):
    """Quantum self‑attention block with optional CNN encoder."""
    def __init__(self, n_wires: int = 4, use_cnn: bool = True):
        super().__init__()
        self.n_wires = n_wires
        self.use_cnn = use_cnn
        if use_cnn:
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        else:
            self.encoder = tq.Identity()
        self.q_layer = self._build_q_layer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def _build_q_layer(self) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
                self.rx0 = tq.RX(has_params=True, trainable=True)
                self.ry0 = tq.RY(has_params=True, trainable=True)
                self.rz0 = tq.RZ(has_params=True, trainable=True)
                self.crx0 = tq.CRX(has_params=True, trainable=True)

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

        return QLayer(n_wires)

    @tq.static_support
    def forward(
        self,
        x: torch.Tensor,
        rotation_params: np.ndarray = None,
        entangle_params: np.ndarray = None,
    ) -> torch.Tensor:
        """
        Compute quantum self‑attention.
        Parameters:
            x: Tensor of shape (B, C, H, W) or (B, N)
            rotation_params, entangle_params: optional NumPy arrays that
            are mapped onto the parameterised gates in the quantum layer.
        Returns:
            Tensor of shape (B, n_wires)
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        if self.use_cnn:
            pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
            self.encoder(qdev, pooled)
        else:
            pooled = x.view(bsz, -1)
            self.encoder(qdev, pooled)
        # Apply external parameters if supplied
        if rotation_params is not None:
            self.q_layer.rx0.params = torch.tensor(rotation_params[:3], dtype=torch.float32, device=x.device)
            self.q_layer.ry0.params = torch.tensor(rotation_params[3:6], dtype=torch.float32, device=x.device)
            self.q_layer.rz0.params = torch.tensor(rotation_params[6:9], dtype=torch.float32, device=x.device)
        if entangle_params is not None:
            self.q_layer.crx0.params = torch.tensor(entangle_params[:1], dtype=torch.float32, device=x.device)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)
