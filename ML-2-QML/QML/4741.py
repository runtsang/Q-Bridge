import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumHybridModel(tq.QuantumModule):
    """QCNN‑style variational circuit that processes classical embeddings."""
    def __init__(self, n_wires: int = 5) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps classical features to qubit amplitudes
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["5x5_ryzxy"])
        # Convolutional sub‑module (RX, RZ and CRX gates)
        self.conv = tq.QuantumModule()
        self.conv.rx = tq.RX(has_params=True, trainable=True)
        self.conv.rz = tq.RZ(has_params=True, trainable=True)
        self.conv.crx = tq.CRX(has_params=True, trainable=True)
        # Pooling sub‑module (CNOT followed by Hadamard)
        self.pool = tq.QuantumModule()
        self.pool.cx = tq.CNOT()
        self.pool.h = tq.H()
        # Measurement and post‑processing
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Classical feature vector of shape (batch, feat_dim).  The first
            ``n_wires`` elements are used as input parameters for the encoder.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        # Simple pooling of the input (average over pairs)
        pooled = F.avg_pool1d(x.unsqueeze(1), kernel_size=2).squeeze(1)
        # Encode classical data
        self.encoder(qdev, pooled)
        # Convolutional layer
        self.conv.rx(qdev)
        self.conv.rz(qdev)
        self.conv.crx(qdev)
        # Pooling layer
        self.pool.cx(qdev)
        self.pool.h(qdev)
        # Measurement and normalization
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QuantumHybridModel"]
