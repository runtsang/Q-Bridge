"""Hybrid quantum‑classical LSTM model with a variational quanvolution front‑end.

The :class:`HybridQLSTM` class below mirrors the interface of its classical
counterpart but replaces the 2×2 convolution with a small two‑qubit quantum
kernel and the LSTM gates with a variational quantum circuit.  The quantum
circuit is implemented using :mod:`torchquantum` and is fully differentiable
so that gradients can be propagated through the entire network.

The design follows the same “convolution → recurrent → classifier” flow
as the classical model, enabling a fair comparison between the two regimes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridQLSTM(nn.Module):
    """Quantum‑enhanced hybrid model: Quanvolution → Quantum LSTM → Linear."""

    class QLayer(tq.QuantumModule):
        """Variational layer that implements a single LSTM gate."""

        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Encode the classical input into rotation angles
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )
            # Trainable rotation parameters
            self.params = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # Entangle neighbouring qubits to capture correlations
            for wire in range(self.n_wires):
                tgt = 0 if wire == self.n_wires - 1 else wire + 1
                tqf.cnot(qdev, wires=[wire, tgt])
            return self.measure(qdev)

    class QuanvolutionFilter(tq.QuantumModule):
        """Two‑qubit quantum kernel applied to each 2×2 image patch."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )
            self.kernel = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            device = x.device
            qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
            x = x.view(bsz, 28, 28)
            patches = []
            for r in range(0, 28, 2):
                for c in range(0, 28, 2):
                    data = torch.stack(
                        [
                            x[:, r, c],
                            x[:, r, c + 1],
                            x[:, r + 1, c],
                            x[:, r + 1, c + 1],
                        ],
                        dim=1,
                    )
                    self.encoder(qdev, data)
                    self.kernel(qdev)
                    patches.append(self.measure(qdev).view(bsz, 4))
            return torch.cat(patches, dim=1)

    class QuantumLSTM(nn.Module):
        """Classical LSTM where each gate is a variational quantum circuit."""

        def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.n_qubits = n_qubits
            # Quantum gates for the four LSTM gates
            self.forget_gate = self.QLayer(n_qubits)
            self.input_gate = self.QLayer(n_qubits)
            self.update_gate = self.QLayer(n_qubits)
            self.output_gate = self.QLayer(n_qubits)
            # Linear projections to the qubit space
            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        def forward(
            self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            hx, cx = self._init_states(inputs, states)
            outputs = []
            for x in inputs.unbind(dim=1):  # iterate over sequence dimension
                combined = torch.cat([x, hx], dim=1)
                f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
                i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
                g = torch.tanh(self.update_gate(self.linear_update(combined)))
                o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outputs.append(hx.unsqueeze(1))
            lstm_out = torch.cat(outputs, dim=1)
            return lstm_out, (hx, cx)

        def _init_states(
            self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if states is not None:
                return states
            batch = inputs.size(0)
            device = inputs.device
            return (
                torch.zeros(batch, self.hidden_dim, device=device),
                torch.zeros(batch, self.hidden_dim, device=device),
            )

    def __init__(self, n_qubits: int, hidden_dim: int, tagset_size: int) -> None:
        super().__init__()
        self.filter = self.QuanvolutionFilter()
        self.lstm = self.QuantumLSTM(
            input_dim=4 * 14 * 14, hidden_dim=hidden_dim, n_qubits=n_qubits
        )
        self.classifier = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, 1, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape ``(batch, seq_len, tagset_size)``.
        """
        batch, seq_len, c, h, w = x.shape
        # Flatten batch and sequence for the quantum filter
        x = x.view(batch * seq_len, c, h, w)
        feats = self.filter(x)  # (batch*seq, 4*14*14)
        feats = feats.view(batch, seq_len, -1)  # (batch, seq, feature_dim)
        lstm_out, _ = self.lstm(feats)
        logits = self.classifier(lstm_out)
        return F.log_softmax(logits, dim=-1)
