import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATHybrid(tq.QuantumModule):
    """A quantum variant of the QuantumNATHybrid model.
    Encodes classical image features into a 4‑wire circuit,
    applies a deep variational layer, and measures in the Pauli‑Z basis.
    The circuit is designed to be trainable end‑to‑end with a classical
    classification head."""

    class VariationalLayer(tq.QuantumModule):
        """A deep variational circuit with parameterised single‑ and two‑qubit gates."""
        def __init__(self, n_wires: int = 4, depth: int = 3):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            # Parameterised single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Parameterised entangling gate
            self.cnot = tq.CNOT

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            for _ in range(self.depth):
                # Apply rotations to all wires
                self.rx(qdev, wires=range(self.n_wires))
                self.ry(qdev, wires=range(self.n_wires))
                self.rz(qdev, wires=range(self.n_wires))
                # Entangle adjacent qubits in a ring
                for i in range(self.n_wires):
                    self.cnot(qdev, wires=[i, (i + 1) % self.n_wires])

    def __init__(self, n_wires: int = 4, num_classes: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.var_layer = self.VariationalLayer(n_wires=n_wires, depth=4)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.classifier = nn.Linear(n_wires, num_classes)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode image, run variational circuit, measure,
        normalise and classify."""
        bsz = x.size(0)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

        # Prepare a 4‑wire feature vector from the image
        pooled = torch.nn.functional.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        # Map each 4‑dim group to a single qubit via the encoder
        self.encoder(qdev, pooled)

        # Variational circuit
        self.var_layer(qdev)

        # Measurement
        out = self.measure(qdev)  # (bsz, n_wires)

        # Classical post‑processing
        out = self.norm(out)
        logits = self.classifier(out)
        return logits

__all__ = ["QuantumNATHybrid"]
