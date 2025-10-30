import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import assemble, transpile
import numpy as np
from typing import Tuple

class QGate(tq.QuantumModule):
    """
    Lightweight quantum gate that maps a classical vector to a
    parametric quantum circuit and returns a measurement vector.
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
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True)
                                     for _ in range(n_wires)])
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

class QLSTMCell(nn.Module):
    """
    Quantum LSTM cell that mixes classical linear projections with quantum
    gate activations. The cell follows the structure of the original QLSTM
    implementation but replaces the quantum block with the lightweight
    QGate defined above.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical linear projections
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum gates for each gate
        self.forget_gate = QGate(n_qubits)
        self.input_gate = QGate(n_qubits)
        self.update_gate = QGate(n_qubits)
        self.output_gate = QGate(n_qubits)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
        i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
        g = torch.tanh(self.update_gate(self.linear_update(combined)))
        o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
        cx_new = f * cx + i * g
        hx_new = o * torch.tanh(cx_new)
        return hx_new, cx_new

class QuantumHybridHead(nn.Module):
    """
    Hybrid head that forwards the LSTM output through a parameterised
    two‑qubit quantum circuit and returns the expectation value of Pauli‑Z.
    """
    def __init__(self, input_dim: int, backend, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.backend = backend
        self.shots = shots
        self.shift = shift
        self.circuit = self._build_circuit(input_dim)

    def _build_circuit(self, n_qubits: int) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        theta = qiskit.circuit.Parameter("theta")
        for q in range(n_qubits):
            qc.ry(theta, q)
        qc.measure_all()
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        angles = x.squeeze().tolist()
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.circuit.parameters[0]: a}
                                         for a in angles])
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        if isinstance(counts, list):
            expect = np.array([self._expectation(c) for c in counts])
        else:
            expect = np.array([self._expectation(counts)])
        return torch.tensor(expect, device=x.device, dtype=x.dtype)

    def _expectation(self, counts: dict) -> float:
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        return np.sum(states * probs)

class UnifiedQLSTMNet(nn.Module):
    """
    Quantum‑enhanced sequence tagger that optionally uses a quantum LSTM cell
    and a quantum hybrid head.  When `n_qubits` is 0 the module falls back
    to a classical LSTM and linear classifier, matching the behaviour of
    the pure ML implementation.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0,
                 quantum_backend=None, shots: int = 1024,
                 shift: float = np.pi / 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.n_qubits = n_qubits
        if n_qubits > 0:
            self.lstm_cell = QLSTMCell(embedding_dim, hidden_dim, n_qubits)
            # quantum hybrid head
            if quantum_backend is None:
                quantum_backend = qiskit.Aer.get_backend("aer_simulator")
            self.hybrid_head = QuantumHybridHead(hidden_dim, quantum_backend,
                                                 shots=shots, shift=shift)
            self.head_linear = nn.Linear(1, tagset_size)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: LongTensor of shape (seq_len, batch)
        Returns:
            log‑probabilities over tagset of shape (seq_len, batch, tagset_size)
        """
        if self.n_qubits > 0:
            embeds = self.embedding(sentence)  # (seq_len, batch, embed_dim)
            seq_len, batch, _ = embeds.size()
            hx = torch.zeros(batch, self.lstm_cell.hidden_dim, device=embeds.device)
            cx = torch.zeros(batch, self.lstm_cell.hidden_dim, device=embeds.device)
            outputs = []
            for t in range(seq_len):
                hx, cx = self.lstm_cell(embeds[t], hx, cx)
                outputs.append(hx.unsqueeze(0))
            lstm_out = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden)
            logits_q = self.hybrid_head(lstm_out.view(-1, lstm_out.size(-1)))  # (seq_len*batch, 1)
            logits_q = self.head_linear(logits_q)  # (seq_len*batch, tagset_size)
            logits_q = logits_q.view(seq_len, batch, -1)
            return F.log_softmax(logits_q, dim=-1)
        else:
            embeds = self.embedding(sentence)  # (seq_len, batch, embed_dim)
            lstm_out, _ = self.lstm(embeds)  # (seq_len, batch, hidden)
            logits = self.hidden2tag(lstm_out)  # (seq_len, batch, tagset_size)
            return F.log_softmax(logits, dim=-1)
