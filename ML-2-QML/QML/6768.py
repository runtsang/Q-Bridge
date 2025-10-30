"""Hybrid LSTM with quantum convolutional layers.

This module implements a quantum‑enhanced sequence‑tagging network.
The recurrent core is a variational LSTM where each gate is realised
by a small quantum circuit.  A second quantum block, a QCNN‑style
convolution, preprocesses the word embeddings before they are fed
into the LSTM.  The model is fully differentiable via parameter‑shift
gradients and can be trained on a quantum simulator or a real device.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

class QuantumConvolutionLayer(tq.QuantumModule):
    """QCNN‑style convolution implemented with variational circuits.

    Each pair of adjacent wires receives a fixed sequence of gates
    followed by a trainable 3‑parameter block, mirroring the
    conv_circuit in the reference QCNN implementation.
    """
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encode the classical input into rotation angles
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        # Three trainable parameters per pair
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(3)) for _ in range(n_wires // 2)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional circuit to a batch of inputs.

        Parameters
        ----------
        x : torch.Tensor
            Shape [seq_len, batch, n_wires] – each row contains the input angles for the wires.
        """
        seq_len, batch, n_wires = x.shape
        # Flatten to process all timesteps in one shot
        x_flat = x.reshape(-1, n_wires)
        qdev = tq.QuantumDevice(n_wires=n_wires, bsz=x_flat.shape[0], device=x_flat.device)
        self.encoder(qdev, x_flat)
        for i in range(0, n_wires - 1, 2):
            p = self.params[i // 2]  # 3 parameters
            # Conv circuit (fixed part)
            tqf.rz(-np.pi / 2, qdev, wires=i + 1)
            tqf.cx(qdev, wires=[i + 1, i])
            # Parametrised part
            tqf.rz(p[0], qdev, wires=i)
            tqf.ry(p[1], qdev, wires=i + 1)
            tqf.cx(qdev, wires=[i, i + 1])
            tqf.ry(p[2], qdev, wires=i + 1)
            tqf.cx(qdev, wires=[i + 1, i])
            tqf.rz(np.pi / 2, qdev, wires=i)
        out = self.measure(qdev)  # shape [batch*seq_len, n_wires]
        return out.reshape(seq_len, batch, n_wires)

class QLSTM(nn.Module):
    """Variational LSTM cell with quantum gates for each gate."""
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

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class HybridQLSTMQCNN(nn.Module):
    """Quantum sequence‑tagger that combines a variational LSTM
    with QCNN‑style convolutional preprocessing.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.n_qubits = n_qubits
        # In the quantum design the embedding dimension must match the qubit count
        assert embedding_dim == n_qubits, "embedding_dim must equal n_qubits for quantum mode"
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.conv = QuantumConvolutionLayer(n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)          # shape: [seq_len, batch, embed]
        conv_out = self.conv(embeds)                    # shape: [seq_len, batch, hidden]
        lstm_out, _ = self.lstm(conv_out)               # shape: [seq_len, batch, hidden]
        tag_logits = self.hidden2tag(lstm_out)           # shape: [seq_len, batch, tagset]
        return F.log_softmax(tag_logits, dim=-1)

__all__ = ["QLSTM", "HybridQLSTMQCNN"]
