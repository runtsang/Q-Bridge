import torch
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn as nn

class QFCModelEnhanced(tq.QuantumModule):
    """Quantum part of the hybrid QFCModelEnhanced.

    Implements a depth‑controlled, parameter‑shared variational circuit
    with configurable entanglement. The circuit prepares a state from
    classical features and measures all qubits in the Pauli‑Z basis.
    """

    class QLayer(tq.QuantumModule):
        """
        Variational layer with depth‑controlled, parameter‑shared gates.
        """
        def __init__(self, depth: int, entanglement: str, n_wires: int):
            super().__init__()
            self.depth = depth
            self.entanglement = entanglement
            self.n_wires = n_wires

            # Parameter‑shared single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

            # Entanglement pattern
            if entanglement == "full":
                self.entanglement_layers = [
                    tq.CNOT(wires=[i, (i + 1) % n_wires]) for i in range(n_wires)
                ]
            elif entanglement == "chain":
                self.entanglement_layers = [
                    tq.CNOT(wires=[i, i + 1]) for i in range(n_wires - 1)
                ]
            else:
                self.entanglement_layers = []

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # Apply depth‑controlled layers
            for _ in range(self.depth):
                for wire in range(self.n_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)
                    self.rz(qdev, wires=wire)
                for layer in self.entanglement_layers:
                    layer(qdev)
            # Optional: add a final layer of single‑qubit rotations for readout
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)

    def __init__(self, depth: int = 3, entanglement: str = "full"):
        super().__init__()
        self.n_wires = 4

        # Classical encoder that maps 16 features to 4 wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])

        # Variational quantum layer
        self.q_layer = self.QLayer(depth=depth, entanglement=entanglement, n_wires=self.n_wires)

        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Batch normalization for the quantum output
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Classical pooling to match encoder input size
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)

        # Apply variational layer
        self.q_layer(qdev)

        # Measurement and normalization
        out = self.measure(qdev)
        return self.norm(out)
