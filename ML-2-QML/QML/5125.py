import torch
import torch.nn as nn
import torchquantum as tq
from typing import Optional

# ----------------- Quantum layer -----------------
class QLayer(tq.QuantumModule):
    """
    Lightweight variational block that mixes a random circuit with trainable
    RX/RY rotations.  Each wire is independently parameterised.
    """
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
        return self.measure(qdev)

# ----------------- Hybrid quantum classifier -----------------
class HybridBinaryClassifier(tq.QuantumModule):
    """
    Quantum‑enhanced binary classifier that accepts a batch of feature vectors,
    encodes them into a quantum state, runs a depth‑controlled variational
    circuit, measures expectation values and maps them to logits.
    """
    def __init__(self,
                 num_features: int,
                 n_wires: int,
                 depth: int = 2,
                 shift: float = 0.0) -> None:
        super().__init__()
        self.num_features = num_features
        self.n_wires = n_wires
        self.depth = depth
        self.shift = shift

        # Encode classical data into a product state of X‑rotations
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.qlayer = QLayer(n_wires)
        self.head = nn.Linear(n_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)

        # Classical‑to‑quantum encoding
        self.encoder(qdev, state_batch)

        # Apply the variational block several times to increase expressivity
        for _ in range(self.depth):
            self.qlayer(qdev)

        # Measurement of all qubits in the Z basis
        features = self.qlayer.measure(qdev)

        # Map raw expectations to logits
        logits = self.head(features) + self.shift

        # Convert to probabilities
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QLayer", "HybridBinaryClassifier"]
