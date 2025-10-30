import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# Quantum libraries
from qiskit import QuantumCircuit, Parameter, execute, Aer
from qiskit.quantum_info import Statevector, Pauli
import pennylane as qml

class QuantumGateLayer(nn.Module):
    """
    Variational quantum gate implemented with Qiskit or Pennylane.
    """
    def __init__(self, n_qubits: int, gate_depth: int = 2, backend: str = "qiskit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.gate_depth = gate_depth
        self.backend = backend
        if self.backend == "qiskit":
            self._build_qiskit_circuit()
        elif self.backend == "pennylane":
            self._build_pennylane_circuit()
        else:
            raise ValueError(f"Unsupported backend '{backend}'")

    def _build_qiskit_circuit(self):
        self.params = [Parameter(f"theta_{i}") for i in range(self.n_qubits * self.gate_depth)]
        self.base_circ = QuantumCircuit(self.n_qubits)
        for depth in range(self.gate_depth):
            for q in range(self.n_qubits):
                self.base_circ.rz(self.params[depth * self.n_qubits + q], q)
            for q in range(self.n_qubits - 1):
                self.base_circ.cx(q, q + 1)

    def _build_pennylane_circuit(self):
        dev = qml.device("default.qubit", wires=self.n_qubits)
        @qml.qnode(dev, interface="torch")
        def circuit(x, params):
            for q in range(self.n_qubits):
                qml.RX(x[q], wires=q)
            for depth in range(self.gate_depth):
                for q in range(self.n_qubits):
                    qml.RZ(params[depth, q], wires=q)
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]
        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if self.backend == "qiskit":
            outputs = []
            for i in range(batch):
                circ = self.base_circ.copy()
                for q in range(self.n_qubits):
                    circ.rx(x[i, q].item(), q)
                bound_params = {p: 0.0 for p in self.params}
                circ = circ.bind_parameters(bound_params)
                state = Statevector.from_instruction(circ)
                exp_vals = []
                for q in range(self.n_qubits):
                    pauli_str = "I" * q + "Z" + "I" * (self.n_qubits - q - 1)
                    pauli = Pauli(pauli_str)
                    exp_val = state.expectation_value(pauli).real
                    exp_vals.append(exp_val)
                outputs.append(exp_vals)
            return torch.tensor(outputs, device=x.device, dtype=x.dtype)
        else:
            return self.circuit(x, torch.randn(self.gate_depth, self.n_qubits, device=x.device))

class QLSTM(nn.Module):
    """
    Hybrid quantum–classical LSTM cell.
    Each gate is a variational quantum circuit followed by a classical linear map.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 gate_depth: int = 2,
                 connectivity: str = "full",
                 dropout: float = 0.0,
                 backend: str = "qiskit"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.gate_depth = gate_depth
        self.connectivity = connectivity
        self.dropout = nn.Dropout(dropout)

        proj_dim = n_qubits
        self.forget_proj = weight_norm(nn.Linear(input_dim + hidden_dim, proj_dim))
        self.input_proj = weight_norm(nn.Linear(input_dim + hidden_dim, proj_dim))
        self.update_proj = weight_norm(nn.Linear(input_dim + hidden_dim, proj_dim))
        self.output_proj = weight_norm(nn.Linear(input_dim + hidden_dim, proj_dim))

        self.forget_gate = QuantumGateLayer(n_qubits, gate_depth, backend)
        self.input_gate  = QuantumGateLayer(n_qubits, gate_depth, backend)
        self.update_gate = QuantumGateLayer(n_qubits, gate_depth, backend)
        self.output_gate = QuantumGateLayer(n_qubits, gate_depth, backend)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def forward(self,
                inputs: torch.Tensor,
                states: tuple | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_proj(combined)))
            i = torch.sigmoid(self.input_gate(self.input_proj(combined)))
            g = torch.tanh(self.update_gate(self.update_proj(combined)))
            o = torch.sigmoid(self.output_gate(self.output_proj(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(self.dropout(hx.unsqueeze(0)))
        return torch.cat(outputs, dim=0), (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical and quantum LSTM.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 gate_depth: int = 2,
                 connectivity: str = "full",
                 dropout: float = 0.0,
                 backend: str = "qiskit"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim,
                              hidden_dim,
                              n_qubits,
                              gate_depth=gate_depth,
                              connectivity=connectivity,
                              dropout=dropout,
                              backend=backend)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence).unsqueeze(0)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds.squeeze(0))
        attn_weights = torch.softmax(torch.sum(lstm_out, dim=2), dim=1).unsqueeze(2)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        tag_logits = self.hidden2tag(context)
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
