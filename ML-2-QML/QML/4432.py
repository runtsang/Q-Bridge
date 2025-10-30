import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import qiskit
from qiskit import Aer

# ------------------------------------------------------------------
# Quantum‑circuit wrapper used by the hybrid head
# ------------------------------------------------------------------
class QuantumCircuitWrapper:
    """Minimal two‑qubit parameterised circuit executed on a Qiskit backend."""
    def __init__(self, n_qubits: int, backend, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        qc = qiskit.QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        qc.barrier()
        qc.ry(self.theta, range(self.n_qubits))
        qc.measure_all()
        return qc

    def run(self, thetas: np.ndarray):
        compiled = qiskit.transpile(self.circuit, self.backend)
        qobj = qiskit.assemble(compiled,
                               shots=self.shots,
                               parameter_binds=[{self.theta: t} for t in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

# ------------------------------------------------------------------
# Hybrid head that delegates to a quantum circuit
# ------------------------------------------------------------------
class HybridHead(nn.Module):
    """Dense head that can optionally use a quantum expectation."""
    def __init__(self, in_features: int, n_qubits: int = 0,
                 backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.n_qubits = n_qubits
        if n_qubits > 0:
            self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
            self.shift = shift
        else:
            self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor):
        if self.n_qubits > 0:
            vals = x.view(x.size(0), -1).cpu().numpy()
            exp = self.circuit.run(vals)
            probs = torch.tensor(exp, dtype=torch.float32, device=x.device)
            return torch.cat([probs.unsqueeze(-1), 1 - probs.unsqueeze(-1)], dim=-1)
        logits = self.linear(x)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

# ------------------------------------------------------------------
# Quantum self‑attention block
# ------------------------------------------------------------------
class QuantumSelfAttention:
    """Self‑attention realised with a small Qiskit circuit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = qiskit.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        qc = qiskit.QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self, backend, rotation_params: np.ndarray,
            entangle_params: np.ndarray, shots: int = 1024):
        qc = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)

# ------------------------------------------------------------------
# Quantum LSTM cell with gate‑based quantum modules
# ------------------------------------------------------------------
class QuantumQLSTM(nn.Module):
    """LSTM cell where each gate is a small quantum circuit."""
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

        def forward(self, x: torch.Tensor):
            qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                    bsz=x.shape[0],
                                    device=x.device)
            self.encoder(qdev, x)
            for w, gate in enumerate(self.params):
                gate(qdev, wires=w)
            for w in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[w, w + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

# ------------------------------------------------------------------
# Unified tagger that supports quantum LSTM, quantum attention,
# and a hybrid quantum head
# ------------------------------------------------------------------
class QLSTMGen111(nn.Module):
    """Sequence tagging model with quantum LSTM layers, optional quantum self‑attention,
    and a hybrid quantum‑dense head."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 use_attention: bool = False):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.use_attention = use_attention
        if use_attention:
            self.attn = QuantumSelfAttention(n_qubits)
        self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits) \
            if n_qubits > 0 else nn.LSTM(embedding_dim, hidden_dim)
        self.head = HybridHead(tagset_size, n_qubits=n_qubits,
                               backend=Aer.get_backend("qasm_simulator"))

    def forward(self,
                sentence: torch.Tensor,
                rotation_params: np.ndarray | None = None,
                entangle_params: np.ndarray | None = None):
        embeds = self.word_embeddings(sentence)
        if self.use_attention:
            if rotation_params is None or entangle_params is None:
                raise ValueError("Quantum attention requires rotation and entangle parameters")
            counts = self.attn.run(Aer.get_backend("qasm_simulator"),
                                  rotation_params, entangle_params)
            # Convert measurement counts to a NumPy array of shape (seq_len, embed_dim)
            # For simplicity, use the average of the binary strings as a feature
            seq_len = embeds.size(0)
            avg_features = np.array([sum(int(k, 2) for k in c.keys()) / len(c)
                                    for c in counts])
            embeds = torch.tensor(avg_features, device=embeds.device,
                                  dtype=embeds.dtype).unsqueeze(1)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        logits = self.head(lstm_out.view(len(sentence), -1))
        return torch.log_softmax(logits, dim=-1)

__all__ = ["QLSTMGen111", "QuantumQLSTM", "HybridHead", "QuantumSelfAttention", "QuantumCircuitWrapper"]
