import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """
    Quantum version of the hybrid model.  The quantum circuit is a
    variational layer that acts on 4 qubits.  A classical encoder
    (identical to the one in the ML implementation) feeds the
    input into the circuit via a GeneralEncoder.  The output is
    the expectation values of Pauli‑Z on all qubits, normalised
    with a BatchNorm1d layer.
    """
    class QLayer(tq.QuantumModule):
        """
        Variational layer with a random circuit followed by
        trainable single‑qubit rotations and a controlled‑RX.
        """
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            # Random layer with 50 ops
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            # Trainable single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Controlled‑RX
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[0, 3])

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Classical encoder identical to ML version
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Normalized 4‑dimensional output of shape (B, 4).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        # Classical preprocessing: average pooling
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Encode into qubits
        self.encoder(qdev, pooled)
        # Variational layer
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
