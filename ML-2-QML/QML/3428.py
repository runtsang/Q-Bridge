"""Quantum‑enhanced hybrid model that merges the Quanvolution filter with a quantum LSTM.

The quantum filter encodes each 2×2 patch into a 4‑qubit register and
processes it through a variational circuit.  The LSTM cell uses
small quantum circuits for each gate, following the structure of
the `QLSTM` seed.  The overall architecture mirrors the classical
variant but replaces the filter and the gates with their quantum
counterparts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["QuanvolutionQLSTMHybrid"]

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """
    Quantum 2×2 patch extractor.  Each patch is encoded on a 4‑qubit register
    and processed by a variational layer that outputs a 4‑dimensional
    measurement vector.  The design follows the original quantum filter
    but uses a trainable variational layer instead of a random layer.
    """
    def __init__(self, n_qubits: int = 4, n_params: int = 8) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.var_layer = tqf.RandomLayer(n_ops=n_params, wires=list(range(n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_qubits, bsz=bsz, device=x.device)
        image = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack([image[:, r, c], image[:, r, c+1],
                                     image[:, r+1, c], image[:, r+1, c+1]],
                                    dim=1)
                self.encoder(qdev, patch)
                self.var_layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuantumQLSTM(tq.QuantumModule):
    """
    Quantum LSTM cell where each gate is a small quantum circuit.
    The cell architecture mirrors the classical QLSTM from the seed.
    """
    class _Gate(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList([tqf.RX(has_params=True, trainable=True)
                                         for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(self.n_qubits, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for w, gate in enumerate(self.params):
                gate(qdev, wires=w)
            for w in range(self.n_qubits):
                tgt = 0 if w == self.n_qubits - 1 else w + 1
                tqf.cnot(qdev, wires=[w, tgt])
            return self.measure(qdev)
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self._Gate(n_qubits)
        self.input_gate = self._Gate(n_qubits)
        self.update = self._Gate(n_qubits)
        self.output = self._Gate(n_qubits)
        self.lin_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_output = nn.Linear(input_dim + hidden_dim, n_qubits)
    def forward(self, seq: torch.Tensor, state: tuple = None) -> tuple:
        hx, cx = self._init_state(seq, state)
        outputs = []
        for t in seq.unbind(dim=1):
            combined = torch.cat([t, hx], dim=1)
            f = torch.sigmoid(self.forget(self.lin_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.lin_input(combined)))
            g = torch.tanh(self.update(self.lin_update(combined)))
            o = torch.sigmoid(self.output(self.lin_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        out_seq = torch.cat(outputs, dim=1)
        return out_seq, (hx, cx)
    def _init_state(self, seq: torch.Tensor, state: tuple = None):
        if state is not None:
            return state
        batch = seq.size(0)
        device = seq.device
        return (torch.zeros(batch, self.hidden_dim, device=device),
                torch.zeros(batch, self.hidden_dim, device=device))

class QuanvolutionQLSTMHybrid(tq.QuantumModule):
    """
    Quantum‑enhanced hybrid that first runs the quanvolution filter
    and then feeds the per‑patch sequence into the quantum LSTM for
    sequence classification/tagging.  The architecture mirrors
    the classical counterpart but replaces the filter and gates
    with their quantum counterparts.
    """
    def __init__(self,
                 num_classes: int = 10,
                 hidden_dim: int = 256,
                 n_qubits: int = 4,
                 n_params: int = 8) -> None:
        super().__init__()
        self.filter = QuantumQuanvolutionFilter(n_qubits=n_qubits, n_params=n_params)
        self.lstm = QuantumQLSTM(self.filter.n_qubits, hidden_dim, n_qubits)
        self.class_head = nn.Linear(hidden_dim, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.filter(x)
        seq = feat.view(x.size(0), -1, self.filter.n_qubits)
        out, _ = self.lstm(seq)
        logits = self.class_head(out[:, -1, :])
        return F.log_softmax(logits, dim=-1)
