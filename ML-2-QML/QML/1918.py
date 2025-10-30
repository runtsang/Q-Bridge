import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torch.quantum import QuantumDevice

class QuantumLayer(tq.QuantumModule):
    """
    Parameter‑shaped Ansatz for 4 qubits:
      * Random layer for initialization
      * RX/RZ on each qubit
      * Controlled rotations for entanglement
    """
    def __init__(self, n_wires=4):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=60, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.rz(qdev, wires=w)
        # entangle adjacent qubits
        for w in range(self.n_wires - 1):
            self.crx(qdev, wires=[w, w + 1])
        # final layer of Hadamard to spread amplitudes
        tqf.hadamard(qdev, wires=range(self.n_wires), static=self.static_mode, parent_graph=self.graph)

class QFCModel(tq.QuantumModule):
    """
    Hybrid quantum‑classical model:
      * Classical CNN backbone
      * Quantum layer that fuses classical features
      * Measurement of Pauli‑Z on all qubits
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires=4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
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

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    def fit(self, train_loader, criterion, optimizer, epochs=10):
        self.train()
        for epoch in range(epochs):
            for batch in train_loader:
                inputs, targets = batch[0].to(x.device), batch[1].to(x.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
