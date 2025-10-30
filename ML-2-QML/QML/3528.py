import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
import torchquantum as tq
import torchquantum.functional as tqf

class HybridFCLLSTM(nn.Module):
    """
    Quantum‑enhanced sequence tagging model that replaces the fully‑connected
    feature mapper and the LSTM gates with parameterised quantum circuits.
    The public API matches the classical version, allowing effortless
    benchmarking across paradigms.  Set ``n_qubits`` to zero to fall back
    to the classical implementation.
    """

    class _FCL(nn.Module):
        """Fallback classical fully‑connected layer."""
        def __init__(self, input_dim: int):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x: torch.Tensor):
            return torch.tanh(self.linear(x))

    class _QuantumFCL(nn.Module):
        """Single‑qubit variational circuit that outputs a scalar expectation."""
        def __init__(self, shots: int = 200):
            super().__init__()
            self.backend = qiskit.Aer.get_backend("qasm_simulator")
            self.shots = shots
            self._circuit = qiskit.QuantumCircuit(1)
            self.theta = qiskit.circuit.Parameter("theta")
            self._circuit.h(0)
            self._circuit.ry(self.theta, 0)
            self._circuit.measure_all()

        def forward(self, thetas: torch.Tensor) -> torch.Tensor:
            # thetas shape: (seq_len, batch, 1)
            seq_len, batch, _ = thetas.shape
            expectations = []
            for theta in thetas.view(-1):
                job = qiskit.execute(
                    self._circuit,
                    self.backend,
                    shots=self.shots,
                    parameter_binds=[{self.theta: float(theta)}],
                )
                result = job.result()
                counts = result.get_counts(self._circuit)
                probs = np.array(list(counts.values())) / self.shots
                states = np.array(list(counts.keys())).astype(float)
                expectations.append(np.sum(states * probs))
            exp_tensor = torch.tensor(expectations, dtype=torch.float32)
            return exp_tensor.view(seq_len, batch, 1)

    class _QuantumQLSTM(tq.QuantumModule):
        """LSTM cell where each gate is a small quantum circuit."""
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

        def __init__(self, input_dim: int, hidden_dim: int, n_wires: int):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.n_wires = n_wires
            self.forget = self.QLayer(n_wires)
            self.input = self.QLayer(n_wires)
            self.update = self.QLayer(n_wires)
            self.output = self.QLayer(n_wires)
            self.linear_f = nn.Linear(input_dim + hidden_dim, n_wires)
            self.linear_i = nn.Linear(input_dim + hidden_dim, n_wires)
            self.linear_g = nn.Linear(input_dim + hidden_dim, n_wires)
            self.linear_o = nn.Linear(input_dim + hidden_dim, n_wires)

        def forward(self, inputs: torch.Tensor, states=None):
            hx, cx = self._init_states(inputs, states)
            outs = []
            for x in inputs.unbind(dim=0):
                combined = torch.cat([x, hx], dim=1)
                f = torch.sigmoid(self.forget(self.linear_f(combined)))
                i = torch.sigmoid(self.input(self.linear_i(combined)))
                g = torch.tanh(self.update(self.linear_g(combined)))
                o = torch.sigmoid(self.output(self.linear_o(combined)))
                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outs.append(hx.unsqueeze(0))
            return torch.cat(outs, dim=0), (hx, cx)

        def _init_states(self, inputs, states):
            if states is not None:
                return states
            batch_size = inputs.size(1)
            device = inputs.device
            return torch.zeros(batch_size, self.hidden_dim, device=device), \
                   torch.zeros(batch_size, self.hidden_dim, device=device)

    def __init__(self, n_features: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 4):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, n_features)
        if n_qubits > 0:
            self.fcl = self._QuantumFCL()
            self.lstm = self._QuantumQLSTM(1, hidden_dim, n_qubits)
        else:
            self.fcl = self._FCL(n_features)
            self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.LongTensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)                 # (seq_len, batch, n_features)
        fcl_out = self.fcl(embeds).squeeze(-1)                  # (seq_len, batch)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(fcl_out.unsqueeze(-1))
        else:
            lstm_out, _ = self.lstm(fcl_out.unsqueeze(-1))
        logits = self.hidden2tag(lstm_out)                      # (seq_len, batch, tagset_size)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridFCLLSTM"]
