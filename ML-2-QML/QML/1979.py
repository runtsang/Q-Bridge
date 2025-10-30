import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as pnp
from typing import Tuple, List, Dict

class QLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM that uses a variational circuit per gate.
    The circuit is a parameter‑shaded sequence of single‑qubit rotations
    followed by entangling CNOTs.  All parameters are differentiable
    and can be optimised with the standard PyTorch autograd.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 gate_families: List[str] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self._gate_families = gate_families or ["rx", "ry", "rz"]

        # Linear maps to qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum circuits – one per gate
        self.forget_circuit = self._make_circuit("forget")
        self.input_circuit = self._make_circuit("input")
        self.update_circuit = self._make_circuit("update")
        self.output_circuit = self._make_circuit("output")

    def _make_circuit(self, name: str):
        n = self.n_qubits
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev, interface="torch")
        def circuit(x, params):
            # Encode the input vector as rotation angles
            for i in range(n):
                qml.RX(x[i], wires=i)
            # Apply trainable continuous rotations
            for i in range(n):
                gate = self._gate_families[i % len(self._gate_families)]
                if gate == "rx":
                    qml.RX(params[i], wires=i)
                elif gate == "ry":
                    qml.RY(params[i], wires=i)
                else:
                    qml.RZ(params[i], wires=i)
            # Entangle
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure in computational basis
            return qml.expval(qml.PauliZ(0))
        # Parameters for the circuit
        params = nn.Parameter(torch.randn(n))
        self.register_parameter(f"{name}_params", params)
        return circuit

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_circuit(self.linear_forget(combined),
                                                   getattr(self, "forget_params")))
            i = torch.sigmoid(self.input_circuit(self.linear_input(combined),
                                                   getattr(self, "input_params")))
            g = torch.tanh(self.update_circuit(self.linear_update(combined),
                                                getattr(self, "update_params")))
            o = torch.sigmoid(self.output_circuit(self.linear_output(combined),
                                                   getattr(self, "output_params")))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def get_gate_params(self) -> Dict[str, torch.Tensor]:
        """Return a dictionary of the trainable parameters for each gate."""
        return {
            "forget": getattr(self, "forget_params"),
            "input": getattr(self, "input_params"),
            "update": getattr(self, "update_params"),
            "output": getattr(self, "output_params"),
        }

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses the hybrid quantum LSTM.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 gate_families: List[str] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits, gate_families)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
