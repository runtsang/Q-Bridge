import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class QLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell where each gate is computed by a
    small variational quantum circuit.  The recurrent core is
    purely classical, so the model can be trained with standard
    back‑propagation while still accessing quantum gradients.
    """
    class _QuantumGate(nn.Module):
        """
        Variational circuit for a single gate.
        Parameters are stored as a trainable tensor of shape
        (n_qubits, depth, 3) representing RX, RY, RZ angles.
        """
        def __init__(self, n_qubits: int, depth: int = 1, device: str = "default.qubit"):
            super().__init__()
            self.n_qubits = n_qubits
            self.depth = depth
            self.device = qml.device(device, wires=n_qubits, shots=0)
            # Parameter tensor: shape (n_qubits, depth, 3)
            self.params = nn.Parameter(torch.randn(n_qubits, depth, 3) * 0.1)

            # JIT‑compile the circuit for speed
            self.circuit = qml.qnode(self._circuit, device=self.device, interface="torch", diff_method="backprop")

        def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # x is a 1‑D tensor of length n_qubits
            for q in range(self.n_qubits):
                qml.RX(x[q], wires=q)
            for d in range(self.depth):
                for q in range(self.n_qubits):
                    qml.RX(params[q, d, 0], wires=q)
                    qml.RY(params[q, d, 1], wires=q)
                    qml.RZ(params[q, d, 2], wires=q)
                # Entangling layer – simple CNOT chain
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            # Return expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(wires=[q])) for q in range(self.n_qubits)]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x is a 1‑D tensor of length n_qubits
            out = self.circuit(x, self.params)
            return out

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, gate_depth: int = 1, device: str = "default.qubit"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.gate_depth = gate_depth

        # Linear projection that feeds the quantum gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum gates
        self.forget_gate = self._QuantumGate(n_qubits, depth=gate_depth, device=device)
        self.input_gate = self._QuantumGate(n_qubits, depth=gate_depth, device=device)
        self.update_gate = self._QuantumGate(n_qubits, depth=gate_depth, device=device)
        self.output_gate = self._QuantumGate(n_qubits, depth=gate_depth, device=device)

        # Final linear mapping from gate outputs to hidden dimension
        self.forget_out = nn.Linear(n_qubits, hidden_dim)
        self.input_out = nn.Linear(n_qubits, hidden_dim)
        self.update_out = nn.Linear(n_qubits, hidden_dim)
        self.output_out = nn.Linear(n_qubits, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_out(self.forget_gate(self.forget_linear(combined))))
            i = torch.sigmoid(self.input_out(self.input_gate(self.input_linear(combined))))
            g = torch.tanh(self.update_out(self.update_gate(self.update_linear(combined))))
            o = torch.sigmoid(self.output_out(self.output_gate(self.output_linear(combined))))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between the quantum
    QLSTM implemented above and the default torch.nn.LSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
