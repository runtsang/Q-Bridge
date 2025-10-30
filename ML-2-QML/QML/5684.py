import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridNATQLSTM(tq.QuantumModule):
    """Quantum hybrid architecture combining a CNN encoder, a quantum fully connected layer,
    and a quantum LSTM for sequence tagging."""
    class QFCBlock(tq.QuantumModule):
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.norm = nn.BatchNorm1d(n_wires)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                    device=x.device, record_op=True)
            self.encoder(qdev, x)
            self.random_layer(qdev)
            out = self.measure(qdev)
            return self.norm(out)

    class QLSTMCell(tq.QuantumModule):
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
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                tgt = 0 if wire == self.n_wires - 1 else wire + 1
                tqf.cnot(qdev, wires=[wire, tgt])
            return self.measure(qdev)

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Classical CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        # Linear reduction to match quantum wire count
        self.cnn_fc = nn.Linear(16 * 7 * 7, n_qubits)
        # Quantum fully connected block
        self.qfc = self.QFCBlock(n_wires=n_qubits)

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Quantum LSTM gates
        self.forget_gate = self.QLSTMCell(n_qubits)
        self.input_gate = self.QLSTMCell(n_qubits)
        self.update_gate = self.QLSTMCell(n_qubits)
        self.output_gate = self.QLSTMCell(n_qubits)

        # Linear projections to quantum wires
        self.linear_forget = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(embedding_dim + hidden_dim, n_qubits)

        # Final classification layer
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: LongTensor of shape (seq_len, batch_size)
            image: FloatTensor of shape (batch_size, 1, H, W)
        Returns:
            Log probabilities over tags.
        """
        # CNN + linear reduction
        img_feat = self.cnn(image)                     # (batch_size, 16*7*7)
        img_reduced = self.cnn_fc(img_feat)            # (batch_size, n_qubits)
        img_out = self.qfc(img_reduced)                # (batch_size, n_qubits)

        # Embedding
        embeds = self.embedding(sentence)              # (seq_len, batch_size, embedding_dim)

        # Initialize hidden and cell states
        batch_size = embeds.size(1)
        hx = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)

        outputs = []
        for x in embeds.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        lstm_out = torch.cat(outputs, dim=0)           # (seq_len, batch_size, hidden_dim)

        tag_logits = self.hidden2tag(lstm_out)         # (seq_len, batch_size, tagset_size)

        # combine image features with tag logits
        tag_logits += img_out.unsqueeze(0)             # broadcast over sequence length

        return F.log_softmax(tag_logits, dim=-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(hidden_dim={self.hidden_dim})"

__all__ = ["HybridNATQLSTM"]
