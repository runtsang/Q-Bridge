import torch
import torch.nn as nn
import torchquantum as tq

class SamplerQNN__gen111(tq.QuantumModule):
    """
    Quantum sampler network with a regression head.
    Encodes a two‑qubit state, applies a parameterised variational layer,
    measures all qubits in the Z basis, and maps the measurement
    statistics to a scalar prediction.
    """

    class QLayer(tq.QuantumModule):
        """
        Variational layer consisting of a random layer followed by
        single‑qubit rotations on each wire.
        """
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, n_wires: int = 2):
        super().__init__()
        # Encoder that maps classical input vectors to quantum states
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Linear head mapping measurement outcomes to a scalar
        self.head = nn.Linear(n_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode the input, run the variational layer,
        measure, and regress.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["SamplerQNN__gen111"]
