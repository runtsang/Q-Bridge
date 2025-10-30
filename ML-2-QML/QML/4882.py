"""Quantum‑enhanced hybrid architecture integrating quanvolution, quantum LSTM,
   quantum self‑attention and a Qiskit EstimatorQNN regression head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator


class QuanvolutionFilter(tq.QuantumModule):
    """Apply a two‑qubit quantum kernel to 2×2 image patches."""
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
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
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
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QLSTM(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
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

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
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

    def forward(self, inputs: torch.Tensor, states: tuple | None = None) -> tuple:
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

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple | None,
    ) -> tuple:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class QuantumSelfAttention(tq.QuantumModule):
    """Quantum circuit that produces an attention vector from a hidden state."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "rz", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        return self.measure(qdev)  # (B, n_wires)


class EstimatorWrapper(nn.Module):
    """Wraps Qiskit's EstimatorQNN to be used as a PyTorch module."""
    def __init__(self, n_qubits: int = 1) -> None:
        super().__init__()
        # Build a simple 1‑qubit regression circuit
        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        observable = SparsePauliOp.from_list([("Y", 1)])
        # Qiskit Estimator
        estimator = Estimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[params[0]],
            weight_params=[params[1]],
            estimator=estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (B, 2) – first two features are used as input/weight
        inputs = x[:, :2].detach().cpu().numpy()
        preds = self.estimator_qnn.predict(inputs)
        preds_np = np.array(preds).reshape(-1, 1)
        return torch.tensor(preds_np, device=x.device, dtype=torch.float32)


class QuanvolutionHybrid(nn.Module):
    """
    Full quantum‑enhanced model that fuses quanvolution, quantum LSTM,
    quantum self‑attention and a Qiskit EstimatorQNN regression head.
    """
    def __init__(self, hidden_dim: int = 32, attention_dim: int = 4, n_qubits: int = 4) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.lstm = QLSTM(input_dim=4, hidden_dim=hidden_dim, n_qubits=n_qubits)
        self.attention = QuantumSelfAttention(n_wires=attention_dim)
        self.estimator = EstimatorWrapper(n_qubits=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Regression predictions of shape (B, 1).
        """
        # 1️⃣ Quantum quanvolution
        features = self.qfilter(x)                      # (B, 4*14*14)
        seq = features.view(x.size(0), 4, 14 * 14).permute(0, 2, 1)  # (B, 196, 4)
        # 2️⃣ Quantum LSTM over patch sequence
        lstm_out, _ = self.lstm(seq)                    # (B, 196, hidden_dim)
        lstm_mean = lstm_out.mean(dim=1)                # (B, hidden_dim)
        # 3️⃣ Quantum self‑attention on the aggregated hidden state
        attn_vec = self.attention(lstm_mean)            # (B, attention_dim)
        # 4️⃣ Quantum regression head
        out = self.estimator(attn_vec)                  # (B, 1)
        return out


__all__ = ["QuanvolutionHybrid"]
