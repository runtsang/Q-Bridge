"""Hybrid quantum architecture combining feature encoding and a parametric sampler.

The model encodes 2‑D inputs into a 4‑qubit state with a GeneralEncoder,
then runs a variational sampler circuit that mixes random gates with a
Ry‑CNOT pattern inspired by the SamplerQNN example.  The output of the
circuit is measured in the Pauli‑Z basis and passed through a batch‑norm
layer to produce four output probabilities.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F

class HybridNatModel(tq.QuantumModule):
    """Quantum module that fuses encoding, variational circuit and sampling."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.sampler_circuit = self.SamplerCircuit()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    class SamplerCircuit(tq.QuantumModule):
        """Parametric sampler circuit combining a random layer and Ry–CNOT pattern."""
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            # Trainable Ry gates on each qubit
            self.rys = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(self.n_wires)])
            # CNOT ladder
            self.cnot_pairs = [(i, (i + 1) % self.n_wires) for i in range(self.n_wires)]

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for i, ry in enumerate(self.rys):
                ry(qdev, wires=i)
            for control, target in self.cnot_pairs:
                tqf.cnot(qdev, wires=[control, target], static=self.static_mode, parent_graph=self.graph)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.sampler_circuit(qdev)
        out = self.measure(qdev)  # shape (bsz, n_wires)
        return self.norm(out)

__all__ = ["HybridNatModel"]
