from __future__ import annotations

import torch
import torch.nn as nn
import torch.quantum as tq  # alias for torchquantum; import as tq
import numpy as np

# Quantum quanvolution filter (identical to the original QML seed, but extended)
class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum replacement of the classical 2×2 filter."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack([x[:, r, c], x[:, r, c + 1],
                                    x[:, r + 1, c], x[:, r + 1, c + 1]],
                                   dim=1)
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)  # (batch, 4*14*14)

class QuantumSelfAttention(tq.QuantumModule):
    """Quantum self‑attention block built with a small variational circuit."""
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "rx", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "rz", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])
        self.q_layer = tq.RandomLayer(n_ops=6, wires=list(range(self.n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_qubits, bsz=bsz, device=device)
        # Use the first n_qubits of the flattened input as amplitudes
        data = x.view(bsz, -1)[:, :self.n_qubits]
        self.encoder(qdev, data)
        self.q_layer(qdev)
        return self.measure(qdev).view(bsz, self.n_qubits)

# Variational quantum auto‑encoder based on a RealAmplitudes ansatz and a swap‑test
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.circuit.library import RealAmplitudes
from qiskit import Aer
from qiskit import execute

class QuantumAutoencoder(tq.QuantumModule):
    """Variational auto‑encoder that compresses a quantum state into 2 classical bits."""
    def __init__(self, latent_dim: int = 3, trash_dim: int = 2) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.num_qubits = latent_dim + 2 * trash_dim + 1
        self.circuit = self._build_circuit()
        self.sampler = SamplerQNN(circuit=self.circuit,
                                  input_params=[],
                                  weight_params=self.circuit.parameters,
                                  interpret=lambda x: x,
                                  output_shape=(2,),
                                  sampler=Aer.get_backend("aer_simulator"))

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        ansatz = RealAmplitudes(self.num_qubits - 1, reps=5)
        circuit.compose(ansatz, range(self.num_qubits - 1), inplace=True)
        aux = self.num_qubits - 1
        circuit.h(aux)
        for i in range(self.trash_dim):
            circuit.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, latent_dim) – not used in this simplified example
        return self.sampler(x)

class QuanvolutionAutoencoderClassifierQNN(nn.Module):
    """Quantum hybrid classifier that chains a quanvolution filter, quantum self‑attention,
    a variational auto‑encoder and a classical linear head."""
    def __init__(self, num_classes: int = 10, latent_dim: int = 3) -> None:
        super().__init__()
        self.qfilter   = QuantumQuanvolutionFilter()
        self.attn      = QuantumSelfAttention()
        self.autoenc   = QuantumAutoencoder(latent_dim=latent_dim)
        self.classifier = nn.Linear(2, num_classes)  # auto‑encoder outputs 2 bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        filt = self.qfilter(x)          # (batch, 4*14*14)
        attn = self.attn(filt)          # (batch, 4)
        latent = self.autoenc(attn)     # (batch, 2)
        logits = self.classifier(latent)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuantumQuanvolutionFilter",
           "QuantumSelfAttention",
           "QuantumAutoencoder",
           "QuanvolutionAutoencoderClassifierQNN"]
