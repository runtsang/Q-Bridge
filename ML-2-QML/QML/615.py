import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """
    Quantum‑augmented variant of the original QFCModel.  It replaces the
    random layer with a compact variational ansatz, performs an
    intermediate measurement to expose partial quantum information,
    and appends a classical linear head.  The output is a 4‑dimensional
    vector that can be jointly trained with the classical branch.
    """

    class VariationalBlock(tq.QuantumModule):
        def __init__(self, n_wires: int = 4, n_layers: int = 2):
            super().__init__()
            self.n_wires = n_wires
            self.n_layers = n_layers
            self.params = nn.Parameter(torch.randn(n_layers, n_wires, 3))

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            for layer in range(self.n_layers):
                for w in range(self.n_wires):
                    # Use a parameterized Ry-Rz-Ry sequence
                    tqf.ry(qdev, w, self.params[layer, w, 0])
                    tqf.rz(qdev, w, self.params[layer, w, 1])
                    tqf.ry(qdev, w, self.params[layer, w, 2])
                # Entangle with CNOT pairs
                for (a, b) in [(0, 1), (2, 3)]:
                    tqf.cnot(qdev, [a, b])

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.var_block = self.VariationalBlock(n_wires=self.n_wires)
        self.mid_measure = tq.MeasureAll(tq.PauliZ)
        self.classical_head = nn.Linear(self.n_wires, 4)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Encode the average‑pooled image features into the quantum device
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)

        # Variational circuit
        self.var_block(qdev)

        # Mid‑layer measurement (optional intermediate readout)
        mid_out = self.mid_measure(qdev)

        # Final measurement
        final_out = self.mid_measure(qdev)

        # Concatenate classical and quantum readouts
        out = torch.cat([mid_out, final_out], dim=1)
        out = self.norm(out)
        out = self.classical_head(out)
        return out


__all__ = ["QuantumNATEnhanced"]
