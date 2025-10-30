"""Quantum‑enhanced model combining CNN, variational refinement, and quantum LSTM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["HybridNATQLSTM"]

class HybridNATQLSTM(nn.Module):
    """Hybrid CNN + quantum refinement + quantum LSTM for sequence modeling."""
    def __init__(self, hidden_dim: int = 128, refine: bool = True):
        super().__init__()
        self.backbone = self._CNNBackbone()
        self.refine = refine
        if refine:
            self.refine_layer = self._QuantumRefine()
        self.lstm = self.QLSTM(4, hidden_dim, n_qubits=4)
        self.output_layer = nn.Linear(hidden_dim, 4)

    class _CNNBackbone(nn.Module):
        """Convolutional feature extractor producing a 4‑dimensional vector."""
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * 7 * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
            )
            self.norm = nn.BatchNorm1d(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.size(0)
            feat = self.features(x)
            flat = feat.view(bsz, -1)
            out = self.fc(flat)
            return self.norm(out)

    class _QuantumRefine(tq.QuantumModule):
        """Variational refinement circuit acting on the 4‑dim vector."""
        def __init__(self, n_wires: int = 4) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.norm = nn.BatchNorm1d(self.n_wires)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (batch, 4).

            Returns
            -------
            torch.Tensor
                Refined vector of shape (batch, 4).
            """
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
            self.encoder(qdev, x)
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.rx(qdev, wires=3)
            out = self.measure(qdev)
            return self.norm(out)

    class QLSTM(nn.Module):
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [
                        {"input_idx": [0], "func": "rx", "wires": [0]},
                        {"input_idx": [1], "func": "rx", "wires": [1]},
                        {"input_idx": [2], "func": "rx", "wires": [2]},
                        {"input_idx": [3], "func": "rx", "wires": [3]},
                    ]
                )
                self.params = nn.ModuleList(
                    [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
                self.encoder(qdev, x)
                for wire, gate in enumerate(self.params):
                    gate(qdev, wires=wire)
                for wire in range(self.n_wires):
                    if wire == self.n_wires - 1:
                        tqf.cnot(qdev, wires=[wire, 0])
                    else:
                        tqf.cnot(qdev, wires=[wire, wire + 1])
                return self.measure(qdev)

        def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.n_qubits = n_qubits

            self.forget = self.QLayer(n_qubits)
            self.input = self.QLayer(n_qubits)
            self.update = self.QLayer(n_qubits)
            self.output = self.QLayer(n_qubits)

            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        def forward(self, inputs: torch.Tensor, states: tuple | None = None):
            hx, cx = self._init_states(inputs, states)
            outputs = []
            for x in inputs.unbind(dim=0):
                combined = torch.cat([x, hx], dim=1)
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output(self.linear_output(combined)))
                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outputs.append(hx.unsqueeze(0))
            outputs = torch.cat(outputs, dim=0)
            return outputs, (hx, cx)

        def _init_states(self, inputs: torch.Tensor, states: tuple | None):
            if states is not None:
                return states
            batch_size = inputs.size(1)
            device = inputs.device
            return (
                torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, seq_len, 4).
        """
        batch, seq_len, *rest = x.shape
        x = x.view(batch * seq_len, *rest)  # (batch*seq_len, 1, 28, 28)
        feats = self.backbone(x)  # (batch*seq_len, 4)
        if self.refine:
            feats = self.refine_layer(feats)  # (batch*seq_len, 4)
        feats = feats.view(batch, seq_len, -1).transpose(0, 1)  # (seq_len, batch, 4)
        lstm_out, _ = self.lstm(feats)  # (seq_len, batch, hidden_dim)
        out = self.output_layer(lstm_out.transpose(0,1))  # (batch, seq_len, 4)
        return out
