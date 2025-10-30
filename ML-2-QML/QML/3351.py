import torch
import torch.nn as nn
import torchquantum as tq

class HybridNAT(tq.QuantumModule):
    """Quantum sub‑module that encodes a high‑dimensional feature vector."""
    def __init__(self, n_wires: int = 4, input_dim: int = 96):
        super().__init__()
        self.n_wires = n_wires
        self.input_dim = input_dim
        # Linear mapping to qubit‑dimension
        self.linear_enc = nn.Linear(input_dim, n_wires)
        # Random layer to enrich expressivity
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
        # Parameterized single‑qubit rotations
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)
        # Photonic‑inspired encoder
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Trainable scaling and shifting (FraudDetection style)
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('shift', torch.tensor(0.0))
        self.norm = nn.BatchNorm1d(n_wires)

    @tq.static_support
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (B, input_dim)
        Returns:
            Tensor of shape (B, n_wires) after measurement, scaling, shifting, and batch‑norm.
        """
        bsz = features.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=features.device,
            record_op=True
        )
        # Encode features into qubit rotations
        enc_params = self.linear_enc(features)
        self.encoder(qdev, enc_params)
        # Random layer for expressivity
        self.random_layer(qdev)
        # Custom gate sequence
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        # Measurement and post‑processing
        out = self.measure(qdev)
        out = out * self.scale + self.shift
        return self.norm(out)
