import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum variant of the hybrid encoder.

    Encodes a 28×28 grayscale image into 8 qubits, applies a
    parameterized variational circuit, and measures Pauli‑Z
    expectation values to produce an 8‑dimensional embedding.
    """

    def __init__(self):
        super().__init__()
        self.n_wires = 8
        # Encoder that maps 8‑dimensional classical data onto 8 qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["8x8_ryzxy"])
        # Linear projection from pixel space to 8 qubits
        self.proj = nn.Linear(28 * 28, self.n_wires)
        # Variational circuit
        self.var_layer = tq.RandomLayer(n_ops=80, wires=list(range(self.n_wires)))
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Post‑processing batch norm
        self.out_bn = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 28, 28)
        Returns: (B, 8) embedding
        """
        bsz = x.shape[0]
        # Flatten pixel values
        flat = x.view(bsz, -1)  # (B, 784)
        # Project to 8 qubits
        features = self.proj(flat)  # (B, 8)
        # Create quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Encode classical data into qubits
        self.encoder(qdev, features)
        # Apply variational circuit
        self.var_layer(qdev)
        # Entangle and add Hadamards
        for i in range(0, self.n_wires, 2):
            tqf.cnot(qdev, wires=[i, i + 1])
        tqf.hadamard(qdev, wires=range(self.n_wires))
        # Measure Pauli‑Z expectation values
        out = self.measure(qdev)
        return self.out_bn(out)

__all__ = ["QuantumNATEnhanced"]
