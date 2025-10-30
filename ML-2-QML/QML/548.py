import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum‑enhanced version of QuantumNATEnhanced.

    The model keeps the same classical encoder as the pure‑classical
    version but replaces the final fully‑connected head with a
    4‑qubit variational circuit.  The depth of the circuit is
    controllable via the ``depth`` argument, enabling a study of
    depth‑performance trade‑offs.  Dropout and batch‑norm are added
    to the measurement outcomes for regularisation.
    """

    class QLayer(tq.QuantumModule):
        """Parameter‑efficient quantum layer with adjustable depth.

        Parameters
        ----------
        depth : int
            Number of times the basic gate block is repeated.
        """

        def __init__(self, depth: int = 1):
            super().__init__()
            self.depth = depth
            self.n_wires = 4
            # Basic variational block: random layer + parameterised rotations
            self.random_layer = tq.RandomLayer(n_ops=10, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cnot = tq.CNOT

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            for _ in range(self.depth):
                self.random_layer(qdev)
                # Apply a small entangling block
                self.rx(qdev, wires=0)
                self.ry(qdev, wires=1)
                self.rz(qdev, wires=2)
                self.cnot(qdev, wires=[0, 1])
                self.cnot(qdev, wires=[2, 3])
                self.rx(qdev, wires=3)
                self.ry(qdev, wires=0)
                self.rz(qdev, wires=1)

    def __init__(self, depth: int = 1, dropout_prob: float = 0.3):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(depth=depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.dropout = nn.Dropout(dropout_prob)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Classical encoder: average‑pool to a 16‑dim vector
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                bsz=bsz,
                                device=x.device,
                                record_op=True)
        # Encode the classical features into qubits
        self.encoder(qdev, pooled)
        # Variational layer
        self.q_layer(qdev)
        # Measurement
        out = self.measure(qdev)
        out = self.dropout(out)
        return self.norm(out)
