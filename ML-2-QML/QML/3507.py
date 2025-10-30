"""HybridNAT: quantum‑enhanced implementation with optional classification or tagging modes.

The class extends the classical `HybridNAT` with quantum variational circuits
for feature encoding and gating.  The classification branch uses a CNN
followed by a variational quantum circuit; the tagging branch uses a
quantum LSTM where each gate is a small quantum circuit.  The module
inherits from `torchquantum.QuantumModule` so it can be trained with
any of the torch‑quantum backends.

The API mirrors the classical version: construct with ``mode`` and
call ``forward`` on the appropriate tensor.  The quantum branches
are activated automatically when ``mode`` is set accordingly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["HybridNAT"]


class HybridNAT(tq.QuantumModule):
    """
    Quantum‑enabled hybrid model supporting classification and sequence tagging.
    """

    class QLayer(tq.QuantumModule):
        """
        Variational layer used in the classification branch.
        Builds upon a random layer followed by a small set of trainable gates.
        """
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    class QuantumLSTM(nn.Module):
        """
        Quantum LSTM cell where each gate is a small quantum circuit.
        Mirrors the `QLSTM` implementation from the QML seed.
        """
        class QGate(tq.QuantumModule):
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
                qdev = tq.QuantumDevice(
                    n_wires=self.n_wires,
                    bsz=x.shape[0],
                    device=x.device,
                )
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

            self.forget_gate = self.QGate(n_qubits)
            self.input_gate = self.QGate(n_qubits)
            self.update_gate = self.QGate(n_qubits)
            self.output_gate = self.QGate(n_qubits)

            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        def forward(self, inputs: torch.Tensor, states=None):
            hx, cx = self._init_states(inputs, states)
            outputs = []
            for x in inputs.unbind(dim=0):
                combined = torch.cat([x, hx], dim=1)
                f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
                i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
                g = torch.tanh(self.update_gate(self.linear_update(combined)))
                o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outputs.append(hx.unsqueeze(0))
            outputs = torch.cat(outputs, dim=0)
            return outputs, (hx, cx)

        def _init_states(self, inputs, states):
            if states is not None:
                return states
            batch_size = inputs.size(1)
            device = inputs.device
            return (
                torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device),
            )

    def __init__(
        self,
        mode: str = "classification",
        *,
        in_channels: int = 1,
        num_classes: int = 4,
        hidden_dim: int = 64,
        embedding_dim: int = 128,
        vocab_size: int = 5000,
        tagset_size: int = 10,
        n_wires: int = 4,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        self.n_wires = n_wires

        if self.mode == "classification":
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.encoder = tq.GeneralEncoder(
                tq.encoder_op_list_name_dict["4x4_ryzxy"]
            )
            self.q_layer = HybridNAT.QLayer(n_wires)
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.fc = nn.Linear(n_wires, num_classes)
            self.norm = nn.BatchNorm1d(num_classes)
        elif self.mode == "tagging":
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = HybridNAT.QuantumLSTM(embedding_dim, hidden_dim, n_qubits=n_wires)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "classification":
            bsz = x.shape[0]
            pooled = F.avg_pool2d(x, 6).view(bsz, 16)
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
            )
            self.encoder(qdev, pooled)
            self.q_layer(qdev)
            out = self.measure(qdev)
            out = self.fc(out)
            return self.norm(out)
        else:  # tagging
            embeds = self.embedding(x)
            lstm_out, _ = self.lstm(embeds)
            logits = self.hidden2tag(lstm_out)
            return F.log_softmax(logits, dim=-1)
