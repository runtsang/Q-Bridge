import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = self.generate_superposition_data(num_wires, samples)

    @staticmethod
    def generate_superposition_data(num_wires: int, samples: int):
        omega_0 = np.zeros(2 ** num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_wires, dtype=complex)
        omega_1[-1] = 1.0

        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2 ** num_wires), dtype=complex)
        for i in range(samples):
            states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        labels = np.sin(2 * thetas) * np.cos(phis)
        return states, labels.astype(np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class UnifiedRegressionLSTM(tq.QuantumModule):
    'Quantum‑classical hybrid regression model that uses a quantum LSTM encoder and a quantum regression head.'

    class QLSTM(tq.QuantumModule):
        'LSTM cell with quantum gates for each gate.'

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

        def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
            hx, cx = self._init_states(inputs, states)
            outputs = []
            for x in inputs.unbind(dim=1):  # seq_len dimension
                combined = torch.cat([x, hx], dim=1)
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output(self.linear_output(combined)))
                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outputs.append(hx.unsqueeze(0))
            lstm_out = torch.cat(outputs, dim=0).permute(1, 0, 2)  # (batch, seq_len, hidden_dim)
            return lstm_out, (hx, cx)

        def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
            if states is not None:
                return states
            batch_size = inputs.size(0)
            device = inputs.device
            return (
                torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device),
            )

    class QRegressionHead(tq.QuantumModule):
        'Quantum regression head that encodes the hidden state, applies a random layer, measures, and feeds the result to a classical linear layer.'

        def __init__(self, hidden_dim: int, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.linear = nn.Linear(hidden_dim, n_qubits)
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_qubits)))
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.head = nn.Linear(n_qubits, 1)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            bsz, seq_len, _ = hidden.shape
            flat_hidden = hidden.reshape(bsz * seq_len, -1)
            encoded = self.linear(flat_hidden)
            qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz * seq_len, device=hidden.device)
            self.encoder(qdev, encoded)
            self.random_layer(qdev)
            features = self.measure(qdev)
            out = self.head(features)
            return out.reshape(bsz, seq_len)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, embedding_dim: int | None = None, vocab_size: int | None = None) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        if embedding_dim is not None and vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = None

        lstm_input_dim = embedding_dim if embedding_dim is not None else input_dim
        self.lstm = self.QLSTM(lstm_input_dim, hidden_dim, n_qubits)
        self.regressor = self.QRegressionHead(hidden_dim, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        'Forward pass: x is (batch, seq_len, input_dim) or (batch, seq_len) if embedding is used.'
        if self.embedding is not None:
            x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.regressor(lstm_out)
        return out
