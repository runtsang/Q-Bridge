import torch
import torch.nn as nn
import torchquantum as tq

class UnifiedHybridLayer(tq.QuantumModule):
    """
    Quantum counterpart of UnifiedHybridLayer.
    Uses a quantum encoder, random layer, and parameterized rotations,
    followed by a linear head.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(num_wires)]
        )
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def run(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Execute the quantum circuit on a batch of input states.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        # Encode input
        self.encoder(qdev, state_batch)
        # Random circuit
        self.random_layer(qdev)
        # Parameterized rotations
        for w in range(self.num_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)
        # Measurement
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)
