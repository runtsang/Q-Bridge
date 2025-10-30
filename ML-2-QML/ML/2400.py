import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNAT(nn.Module):
    """
    Hybrid classical‑quantum model that merges:
    * 2‑D convolutional feature extractor (like QFCModel)
    * Fully‑connected head producing 4‑dimensional output
    * QCNN‑style quantum layer (convolution + pooling) implemented with torchquantum
    The classical and quantum outputs are summed element‑wise after batch‑norm.
    """

    class QLayer(tq.QuantumModule):
        """QCNN‑style quantum block: conv + pool with trainable parameters."""
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            # Convolutional parameters
            self.conv_params = tq.ParameterVector("conv", n_wires * 3)
            # Pooling parameters
            self.pool_params = tq.ParameterVector("pool", n_wires * 3)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # Convolution: apply 3‑parameter unitary on each pair of wires
            for i in range(0, self.n_wires, 2):
                idx = i // 2 * 3
                self._conv_unitary(qdev, [i, i + 1], self.conv_params[idx:idx + 3])

            # Pooling: discard every second qubit by measuring and resetting
            for i in range(0, self.n_wires, 2):
                idx = i // 2 * 3
                self._pool_unitary(qdev, [i, i + 1], self.pool_params[idx:idx + 3])

        def _conv_unitary(self, qdev, wires, params):
            """Apply a 3‑parameter rotation sequence on a pair of qubits."""
            qdev.rz(params[0], wires[0])
            qdev.ry(params[1], wires[1])
            qdev.cx(wires[1], wires[0])

        def _pool_unitary(self, qdev, wires, params):
            """Apply a 3‑parameter sequence followed by measurement."""
            qdev.rz(params[0], wires[0])
            qdev.ry(params[1], wires[1])
            qdev.cx(wires[0], wires[1])
            # measurement will be performed outside

    def __init__(self) -> None:
        super().__init__()
        # Classical CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classical FC head
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

        # Quantum encoder and layer
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Classical path
        features = self.features(x)
        flat = features.view(bsz, -1)
        classical_out = self.norm(self.fc(flat))

        # Prepare quantum device
        qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Encode pooled features into quantum state
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)

        # Quantum processing
        self.q_layer(qdev)
        quantum_out = self.norm(self.measure(qdev))

        # Merge classical and quantum signals
        return classical_out + quantum_out
