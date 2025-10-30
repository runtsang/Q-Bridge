"""Hybrid LSTMTagger that uses quantum LSTM gates and a quantum quanvolution filter.

The implementation re‑uses the QLSTM cell from the original quantum example
and replaces the classical convolution with a small two‑qubit kernel.
Self‑attention is kept classical because the quantum self‑attention example
is a pure simulation that does not add trainable parameters.
The public API matches the classical version so the module can be swapped
in without changing downstream code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QLSTM(nn.Module):
    """LSTM cell whose gates are implemented by tiny quantum circuits."""
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

        self.forget  = self.QLayer(n_qubits)
        self.input   = self.QLayer(n_qubits)
        self.update  = self.QLayer(n_qubits)
        self.output  = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input  = nn.Linear(input_dim + hidden_dim, n_qubits)
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
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

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

class QuantumConvFilter(tq.QuantumModule):
    """2×2 patch quantum kernel that mimics the original quanvolution filter."""
    def __init__(self, n_wires: int = 4, n_ops: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        # extract 2×2 patches from 28×28 images
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, 0, r, c],
                        x[:, 0, r, c + 1],
                        x[:, 0, r + 1, c],
                        x[:, 0, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class ClassicalSelfAttention(nn.Module):
    """Scaled dot‑product self‑attention retained from the classical reference."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q_t = q.transpose(0, 1)
        k_t = k.transpose(0, 1)
        v_t = v.transpose(0, 1)
        scores = torch.softmax(
            q_t @ k_t.transpose(-2, -1) / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32)),
            dim=-1
        )
        out = scores @ v_t
        return out.transpose(0, 1)

class LSTMTagger(nn.Module):
    """Hybrid quantum/classical sequence‑tagger that uses a QLSTM cell
    and a quantum quanvolution filter, followed by classical self‑attention."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.conv = QuantumConvFilter()
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.attention = ClassicalSelfAttention(hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # sentence: [seq_len, batch, 1, H, W]
        embeds = self.word_embeddings(sentence)
        seq_len, batch, *_ = embeds.shape
        features = []
        for i in range(seq_len):
            img = sentence[i]  # [batch, 1, H, W]
            feat = self.conv(img)
            features.append(feat)
        conv_out = torch.stack(features, dim=0)  # [seq_len, batch, conv_dim]
        lstm_out, _ = self.lstm(conv_out)
        att_out = self.attention(lstm_out)
        logits = self.hidden2tag(att_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["LSTMTagger"]
