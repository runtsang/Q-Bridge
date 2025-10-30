import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridFCL_QLSTM(nn.Module):
    """
    Hybrid module combining a quantum fully connected layer and a quantum LSTM.
    Supports pure quantum operation mode.
    """
    class _QuantumFCL(tq.QuantumModule):
        """
        Variational circuit for a single qubit fully connected layer.
        """
        def __init__(self, n_wires: int = 1, shots: int = 100):
            super().__init__()
            self.n_wires = n_wires
            self.shots = shots
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "h", "wires": [0]},
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                ]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Use only the first element of each embedding as the parameter
            param = x[:, :1]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=param.shape[0], device=param.device)
            self.encoder(qdev, param)
            return self.measure(qdev)

    class QLayer(tq.QuantumModule):
        """
        Quantum layer used as a gate in the LSTM cell.
        """
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

    class _QuantumLSTM(nn.Module):
        """
        Quantum-enhanced LSTM cell where gates are realized by small quantum circuits.
        """
        def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.n_qubits = n_qubits

            self.forget = HybridFCL_QLSTM.QLayer(n_qubits)
            self.input = HybridFCL_QLSTM.QLayer(n_qubits)
            self.update = HybridFCL_QLSTM.QLayer(n_qubits)
            self.output = HybridFCL_QLSTM.QLayer(n_qubits)

            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        def forward(self, inputs: torch.Tensor, states=None):
            hx, cx = self._init_states(inputs, states)
            outputs = []
            for x in inputs.unbind(dim=1):
                combined = torch.cat([x, hx], dim=1)
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output(self.linear_output(combined)))
                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outputs.append(hx.unsqueeze(1))
            outputs = torch.cat(outputs, dim=1)
            return outputs, (hx, cx)

        def _init_states(self, inputs: torch.Tensor, states=None):
            if states is not None:
                return states
            batch_size = inputs.size(0)
            device = inputs.device
            return (
                torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device),
            )

    def __init__(self, n_features: int, hidden_dim: int, vocab_size: int, tagset_size: int,
                 n_qubits_fcl: int = 1, n_qubits_lstm: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_features)
        self.fcl = self._QuantumFCL(n_wires=n_qubits_fcl)
        self.lstm = self._QuantumLSTM(n_features, hidden_dim, n_qubits_lstm)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        """
        sentence: (batch_size, seq_len) LongTensor of token indices
        """
        embeds = self.embedding(sentence)  # (batch_size, seq_len, n_features)
        # Apply quantum fully connected layer to each timestep
        flat_embeds = embeds.reshape(-1, embeds.size(-1))
        fcl_out = self.fcl(flat_embeds)  # (batch_size*seq_len, n_qubits_fcl)
        fcl_out = fcl_out.reshape(embeds.size(0), embeds.size(1), -1)
        # Pass through quantum LSTM
        lstm_out, _ = self.lstm(fcl_out, None)
        # Output projection
        tag_logits = self.hidden2tag(lstm_out)  # (batch_size, seq_len, tagset_size)
        return F.log_softmax(tag_logits, dim=-1)

__all__ = ["HybridFCL_QLSTM"]
