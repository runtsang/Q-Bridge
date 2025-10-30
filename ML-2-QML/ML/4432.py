import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile

# ------------------------------------------------------------------
# Quantum‑circuit wrapper used by the hybrid head
# ------------------------------------------------------------------
class QuantumCircuitWrapper:
    """A minimal parametrised two‑qubit circuit executed on a Qiskit backend."""
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
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled,
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
# Hybrid dense‑to‑quantum head
# ------------------------------------------------------------------
class HybridHead(nn.Module):
    """Dense head that can optionally delegate to a quantum circuit."""
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
# Classical self‑attention helper
# ------------------------------------------------------------------
class SelfAttentionModule:
    """Simple self‑attention that operates on NumPy arrays."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray):
        query = torch.tensor(inputs @ rotation_params.reshape(self.embed_dim, -1),
                             dtype=torch.float32)
        key = torch.tensor(inputs @ entangle_params.reshape(self.embed_dim, -1),
                           dtype=torch.float32)
        value = torch.tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# ------------------------------------------------------------------
# Classical LSTM cell (drop‑in replacement)
# ------------------------------------------------------------------
class QLSTM(nn.Module):
    """Linear‑gated LSTM cell that mirrors the interface of a quantum LSTM."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
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
# Unified tagger that can switch between classical and quantum LSTM
# ------------------------------------------------------------------
class QLSTMGen111(nn.Module):
    """Sequence tagging model that supports classical or quantum LSTM layers,
    optional self‑attention, and a hybrid classification head."""
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
            self.attn = SelfAttentionModule(embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits) \
            if n_qubits > 0 else nn.LSTM(embedding_dim, hidden_dim)
        self.head = HybridHead(tagset_size, n_qubits=n_qubits)

    def forward(self,
                sentence: torch.Tensor,
                rotation_params: np.ndarray | None = None,
                entangle_params: np.ndarray | None = None):
        embeds = self.word_embeddings(sentence)
        if self.use_attention:
            if rotation_params is None or entangle_params is None:
                raise ValueError("Attention requires rotation and entangle parameters")
            attn_out = self.attn.run(rotation_params, entangle_params,
                                     embeds.detach().cpu().numpy())
            embeds = torch.tensor(attn_out, device=embeds.device, dtype=embeds.dtype)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        logits = self.head(lstm_out.view(len(sentence), -1))
        return torch.log_softmax(logits, dim=-1)

__all__ = ["QLSTMGen111", "QLSTM", "HybridHead", "SelfAttentionModule", "QuantumCircuitWrapper"]
