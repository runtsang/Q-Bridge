import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumGate(nn.Module):
    """
    Quantum variational gate that produces a scalar via a small Pennylane circuit.
    The circuit has a single variational parameter per qubit and a fixed entangling pattern.
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x, theta):
            # Encode input into rotation angles
            for i in range(n_qubits):
                qml.RX(x[i], wires=i)
            # Variational layer
            for i in range(n_qubits):
                qml.RY(theta[i], wires=i)
            # Entangling layer
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation of PauliZ on first qubit
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit
        self.theta = nn.Parameter(torch.rand(n_qubits))

    def forward(self, x: torch.Tensor):
        # x is expected to be of shape (batch, n_qubits)
        return self.circuit(x, self.theta).unsqueeze(-1)

class QuantumLSTMCell(nn.Module):
    """
    LSTM cell where each gate is implemented by a QuantumGate.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.f_gate = QuantumGate(n_qubits)
        self.i_gate = QuantumGate(n_qubits)
        self.g_gate = QuantumGate(n_qubits)
        self.o_gate = QuantumGate(n_qubits)

        # Linear transformations to produce inputs for the gates
        self.f_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.i_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.g_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.o_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor):
        combined = torch.cat([x, hx], dim=1)

        f = torch.sigmoid(self.f_gate(self.f_lin(combined)))
        i = torch.sigmoid(self.i_gate(self.i_lin(combined)))
        g = torch.tanh(self.g_gate(self.g_lin(combined)))
        o = torch.sigmoid(self.o_gate(self.o_lin(combined)))

        new_c = f * cx + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c

class QuantumLSTM(nn.Module):
    """
    Wrapper around QuantumLSTMCell to process a full sequence.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.cell = QuantumLSTMCell(input_dim, hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor = None, c0: torch.Tensor = None):
        batch_size = inputs.size(1)
        device = inputs.device
        if h0 is None:
            h0 = torch.zeros(batch_size, self.cell.hidden_dim, device=device)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.cell.hidden_dim, device=device)

        hx, cx = h0, c0
        outputs = []
        for step in range(inputs.size(0)):
            hx, cx = self.cell(inputs[step], hx, cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

class QuantumTagger(nn.Module):
    """
    Sequence tagging model that switches between the quantum LSTM and
    the classical LSTM.  The interface is identical to the original
    seed.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QuantumLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_to_tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden_to_tag(lstm_out)
        return F.log_softmax(logits, dim=1)

def validate_gate_consistency():
    """
    Simple validation that ensures the quantum gate outputs
    lie in the expected range and produce consistent shapes.
    """
    n_qubits = 4
    gate = QuantumGate(n_qubits)
    x = torch.randn(3, n_qubits)
    out = gate(x)
    assert out.shape == (3, 1), "Output shape mismatch"
    assert torch.all(out >= -1) and torch.all(out <= 1), "Gate output out of bounds"

if __name__ == "__main__":
    validate_gate_consistency()
    print("Quantum gate validation passed.")

__all__ = ["QuantumGate", "QuantumLSTMCell", "QuantumLSTM", "QuantumTagger"]
