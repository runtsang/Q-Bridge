from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# QRNN quantum utilities
from.QRNN import feedforward, backward, random_network

# Quantum classifier builder
from.QuantumClassifierModel import build_classifier_circuit

# Quantum fully‑connected layer
from.FCL import FCL


class SharedClassName(nn.Module):
    """Hybrid LSTM with quantum gates, QRNN utilities, and a quantum classifier head."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 classifier_depth: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.n_qubits = n_qubits
        if n_qubits > 0:
            self.forget = self.QLayer(n_qubits)
            self.input = self.QLayer(n_qubits)
            self.update = self.QLayer(n_qubits)
            self.output = self.QLayer(n_qubits)
            self.linear_forget = nn.Linear(embedding_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(embedding_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(embedding_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Quantum classifier circuit
        self.classifier, self.enc, self.wts, self.obs = build_classifier_circuit(
            num_qubits=n_qubits,
            depth=classifier_depth
        )

        self.fcl = FCL()

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence).unsqueeze(1)
        if self.n_qubits > 0:
            hx = torch.zeros(sentence.size(0), self.hidden_dim, device=embeds.device)
            cx = torch.zeros(sentence.size(0), self.hidden_dim, device=embeds.device)
            outputs = []
            for x in embeds.squeeze(1).unbind(0):
                combined = torch.cat([x, hx], dim=1)
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output(self.linear_output(combined)))
                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outputs.append(hx.unsqueeze(0))
            lstm_out = torch.cat(outputs, dim=0)
        else:
            lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(tag_logits, dim=1)

    def qnn_backward(self, training_data, unitaries, arch):
        """Demonstrate QRNN backward propagation for a quantum network."""
        return backward(training_data, unitaries, arch)

    def fcl_run(self, thetas):
        return self.fcl.run(thetas)


__all__ = ["SharedClassName"]
