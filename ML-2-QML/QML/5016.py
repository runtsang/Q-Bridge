import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchquantum as tq
import torchquantum.functional as tqf

class HybridNAT(tq.QuantumModule):
    """Hybrid quantum‑classical network that retains the classical CNN
    backbone and replaces the final head with a variational quantum circuit.
    The module also exposes a quantum kernel and a tiny EstimatorQNN for
    regression experiments.
    """

    class QLayer(tq.QuantumModule):
        """Variational layer combining random gates and trainable rotations."""
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            # Trainable single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=3)
            self.crx(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    class QuantumKernel(tq.QuantumModule):
        """Simple quantum kernel that encodes two inputs via RX gates."""
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.qdev = tq.QuantumDevice(n_wires=n_wires)

        @tq.static_support
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # x, y are 1‑D tensors of length n_wires
            self.qdev.reset_states(1)
            for i in range(self.n_wires):
                tqf.rx(self.qdev, wires=i, params=x[i])
            for i in range(self.n_wires):
                tqf.rx(self.qdev, wires=i, params=-y[i])
            # Return overlap with |0...0>
            return torch.abs(self.qdev.states.view(-1)[0])

    class EstimatorQNN(tq.QuantumModule):
        """A minimal quantum neural network for regression using a single qubit."""
        def __init__(self):
            super().__init__()
            self.n_wires = 1

        @tq.static_support
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            # inputs shape: (batch, 1)
            bsz = inputs.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=inputs.device, record_op=True)
            tqf.h(qdev, wires=0)
            tqf.ry(qdev, wires=0, params=inputs.squeeze())
            tqf.rx(qdev, wires=0, params=inputs.squeeze())
            return tqf.expectation(qdev, tq.PauliZ)

    def __init__(self,
                 mode: str = "classification",
                 use_quantum_head: bool = True,
                 n_wires: int = 4,
                 shift: float = 0.0):
        super().__init__()
        self.mode = mode
        self.use_quantum_head = use_quantum_head
        self.shift = shift
        self.n_wires = n_wires

        # Classical backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

        # Quantum hybrid head
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Quantum kernel and estimator
        self.kernel = self.QuantumKernel()
        self.estimator = self.EstimatorQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        out = self.fc(flat)
        out = self.norm(out)

        if self.mode == "regression":
            # Use the quantum estimator on the first feature
            reg_inputs = flat[:, :1]
            return self.estimator(reg_inputs)

        if self.use_quantum_head:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
            pooled = F.avg_pool2d(x, 6).view(bsz, 16)
            self.encoder(qdev, pooled)
            self.q_layer(qdev)
            expectation = self.measure(qdev)
            return torch.sigmoid(expectation + self.shift)

        # Classical head
        logits = out[:, 0].unsqueeze(-1)
        return torch.sigmoid(logits + self.shift)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Compute the Gram matrix using the quantum kernel."""
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridNAT"]
