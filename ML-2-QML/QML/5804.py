import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np

class QuantumLayer(tq.QuantumModule):
    """Random‑layer + RX/RY ansatz that produces a feature vector of size `n_wires`."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        # Random layer with 30 gates per wire
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

class QuantumHybrid(nn.Module):
    """Quantum head that encodes a scalar, runs a variational circuit, measures Pauli‑Z, and maps to a logit."""
    def __init__(self, n_wires: int, shift: float = np.pi / 2):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.q_layer = QuantumLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 1)
        self.shift = shift

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode the scalar as a single‑parameter rotation
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        logits = self.head(features).squeeze(-1)
        # Apply a shift‑sigmoid to emulate a quantum expectation output
        probs = torch.sigmoid(logits + self.shift)
        return probs

class QCNetQuantum(nn.Module):
    """CNN backbone followed by a fully quantum hybrid head for binary classification."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Quantum head operates on a single scalar output
        self.quantum_head = QuantumHybrid(n_wires=1, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.quantum_head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumLayer", "QuantumHybrid", "QCNetQuantum"]
