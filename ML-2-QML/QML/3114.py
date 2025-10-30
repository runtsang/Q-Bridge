import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class FeatureExtractor(nn.Module):
    """CNN feature extractor producing 4‑dimensional embeddings."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flattened = feats.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

class QLayer(tq.QuantumModule):
    """Variational layer with trainable single‑qubit rotations and a controlled‑RX."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
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

class QuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel that encodes CNN‑derived features into a 4‑qubit system."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.feature_extractor = FeatureExtractor()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = QLayer()
        self.norm = nn.BatchNorm1d(self.n_wires)

    def _encode_batch(self, x: torch.Tensor, qdev: tq.QuantumDevice):
        """Encode a batch of images into the quantum device."""
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(x.shape[0], 16)
        self.encoder(qdev, pooled)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the quantum kernel matrix between two batches of images."""
        bsz_x = x.shape[0]
        bsz_y = y.shape[0]
        qdev_x = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz_x, device=x.device, record_op=True)
        self._encode_batch(x, qdev_x)
        self.q_layer(qdev_x)
        state_x = qdev_x.states.view(bsz_x, -1)

        qdev_y = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz_y, device=y.device, record_op=True)
        self._encode_batch(y, qdev_y)
        self.q_layer(qdev_y)
        state_y = qdev_y.states.view(bsz_y, -1)

        kernel = torch.abs(torch.mm(state_x, state_y.t()))  # (bsz_x, bsz_y)
        return self.norm(kernel)

    def kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """Convenience wrapper returning a NumPy array."""
        return self.forward(x, y).cpu().numpy()

__all__ = ["FeatureExtractor", "QLayer", "QuantumKernelMethod"]
