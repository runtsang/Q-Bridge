"""Quantum‑centric self‑attention module with classification."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.providers import Backend
import torch
import torch.nn as nn
from typing import Tuple, List

class QuantumSelfAttentionCircuit:
    """Variational circuit that outputs an attention distribution over n qubits."""
    def __init__(self, n_qubits: int, depth: int = 1, backend: Backend | None = None):
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.params = np.random.rand(n_qubits * depth * 3)
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits)
        cr = ClassicalRegister(self.n_qubits)
        qc = QuantumCircuit(qr, cr)
        idx = 0
        for d in range(self.depth):
            for i in range(self.n_qubits):
                rx = self.params[idx]
                ry = self.params[idx+1]
                rz = self.params[idx+2]
                qc.rx(rx, i)
                qc.ry(ry, i)
                qc.rz(rz, i)
                idx += 3
            for i in range(self.n_qubits-1):
                qc.cx(i, i+1)
        qc.measure(qr, cr)
        return qc

    def run(self, shots: int = 1024) -> np.ndarray:
        job = execute(self.circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        probs = np.zeros(self.n_qubits)
        for bitstring, cnt in counts.items():
            idx = int(bitstring[::-1], 2)
            probs[idx] = cnt / shots
        return probs

class QuantumClassifierAnsatz:
    """Variational circuit that produces logits for classification."""
    def __init__(self, n_qubits: int, n_classes: int, depth: int = 1, backend: Backend | None = None):
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.depth = depth
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.params = np.random.rand(n_qubits * depth * 3 + n_classes)
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits)
        cr = ClassicalRegister(self.n_qubits)
        qc = QuantumCircuit(qr, cr)
        idx = 0
        for d in range(self.depth):
            for i in range(self.n_qubits):
                rx = self.params[idx]
                ry = self.params[idx+1]
                rz = self.params[idx+2]
                qc.rx(rx, i)
                qc.ry(ry, i)
                qc.rz(rz, i)
                idx += 3
            for i in range(self.n_qubits-1):
                qc.cx(i, i+1)
        self.biases = self.params[idx:idx+self.n_classes]
        qc.measure(qr, cr)
        return qc

    def run(self, shots: int = 1024) -> np.ndarray:
        job = execute(self.circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        logits = np.zeros(self.n_classes)
        for bitstring, cnt in counts.items():
            idx = int(bitstring[::-1], 2)
            cls = idx % self.n_classes
            logits[cls] += cnt / shots
        return logits + self.biases

class QuantumLSTMCell:
    """Quantum LSTM cell using small circuits to realize gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, backend: Backend | None = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        import torch
        import torch.nn as nn
        self.fc_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.fc_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.fc_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.fc_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _gate_circuit(self, params: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits)
        qc = QuantumCircuit(qr)
        for i, theta in enumerate(params):
            qc.rx(theta, i)
        for i in range(self.n_qubits-1):
            qc.cx(i, i+1)
        return qc

    def _expect_z(self, circ: QuantumCircuit) -> float:
        circ.save_expectation_value(qiskit.circuit.library.Pauli('Z'), [0])
        job = execute(circ, self.backend, shots=1024)
        result = job.result()
        return result.get_expectation_value('PauliExpectation')

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        import torch
        combined = torch.cat([x, h], dim=1)
        f_params = self.fc_forget(combined).detach().cpu().numpy()
        i_params = self.fc_input(combined).detach().cpu().numpy()
        u_params = self.fc_update(combined).detach().cpu().numpy()
        o_params = self.fc_output(combined).detach().cpu().numpy()
        f_circ = self._gate_circuit(f_params)
        i_circ = self._gate_circuit(i_params)
        u_circ = self._gate_circuit(u_params)
        o_circ = self._gate_circuit(o_params)
        f_val = self._expect_z(f_circ)
        i_val = self._expect_z(i_circ)
        u_val = self._expect_z(u_circ)
        o_val = self._expect_z(o_circ)
        f_t = torch.sigmoid(torch.tensor(f_val, dtype=torch.float32))
        i_t = torch.sigmoid(torch.tensor(i_val, dtype=torch.float32))
        u_t = torch.tanh(torch.tensor(u_val, dtype=torch.float32))
        o_t = torch.sigmoid(torch.tensor(o_val, dtype=torch.float32))
        c_new = f_t * c + i_t * u_t
        h_new = o_t * torch.tanh(c_new)
        return h_new, c_new

class QuantumLSTMTagger:
    """Sequence tagging model that uses the quantum LSTM cell."""
    def __init__(self, embed_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int):
        import torch.nn as nn
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = QuantumLSTMCell(embed_dim, hidden_dim, n_qubits)
        self.decoder = nn.Linear(hidden_dim, tagset_size)

    def forward(self, seq: torch.Tensor):
        import torch
        import torch.nn.functional as F
        embeds = self.embed(seq)
        batch_size, seq_len, _ = embeds.shape
        h = torch.zeros(batch_size, self.lstm.hidden_dim)
        c = torch.zeros(batch_size, self.lstm.hidden_dim)
        outputs = []
        for t in range(seq_len):
            h, c = self.lstm(embeds[:, t, :], h, c)
            outputs.append(h.unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        logits = self.decoder(out)
        return F.log_softmax(logits, dim=-1)

class HybridSelfAttention(nn.Module):
    """Quantum‑centric hybrid module combining attention and classification."""
    def __init__(self, n_qubits: int, depth: int = 1, backend=None, n_classes: int = 2):
        super().__init__()
        self.attention = QuantumSelfAttentionCircuit(n_qubits, depth, backend=backend)
        self.classifier = QuantumClassifierAnsatz(n_qubits, n_classes, depth, backend=backend)

    def forward(self, shots: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        attn_weights = self.attention.run(shots=shots)
        logits = self.classifier.run(shots=shots)
        return attn_weights, logits
