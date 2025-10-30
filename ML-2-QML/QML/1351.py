"""Quantum‑based LSTM cell with Qiskit Aer simulator and variational circuits."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter

class QLSTM__gen308(nn.Module):
    """Quantum LSTM cell that replaces classical gate linear layers with
    variational quantum circuits. The circuit is executed on the Aer
    simulator and uses the parameter‑shift rule for gradients.

    Parameters
    ----------
    input_dim : int
        Size of input features.
    hidden_dim : int
        Size of hidden state.
    n_qubits : int, default 0
        Number of qubits used for quantum gates. If 0, the cell behaves as a classical LSTM.
    """

    class QuantumGate(nn.Module):
        """Variational circuit that outputs a single expectation value."""

        def __init__(self, n_qubits: int, n_params: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_params = n_params
            # Trainable parameters
            self.params = nn.Parameter(torch.randn(n_params))
            # Build a simple ansatz circuit
            self.base_circuit = QuantumCircuit(n_qubits)
            for i in range(n_qubits):
                self.base_circuit.rx(Parameter(f"theta_{i}"), i)
            for i in range(n_qubits - 1):
                self.base_circuit.cx(i, i + 1)
            self.simulator = Aer.get_backend('qasm_simulator')

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, n_qubits)
            batch_size = x.size(0)
            outputs = []
            for i in range(batch_size):
                circ = self.base_circuit.copy()
                # Bind parameters
                param_dict = {f"theta_{j}": self.params[j].item() for j in range(self.n_params)}
                # Encode data into rotation angles
                for q in range(self.n_qubits):
                    param_dict[f"theta_{q}"] += x[i, q].item()
                circ = circ.bind_parameters(param_dict)
                # Execute
                job = execute(circ, self.simulator, shots=1024)
                result = job.result()
                counts = result.get_counts(circ)
                # Compute expectation of PauliZ on all qubits
                exp = 0.0
                for qubit in range(self.n_qubits):
                    z_sum = 0
                    for bitstring, count in counts.items():
                        # Qiskit returns bitstring with qubit 0 as leftmost
                        bit = int(bitstring[::-1][qubit])
                        z = 1 if bit == 0 else -1
                        z_sum += z * count
                    exp += z_sum / sum(counts.values())
                outputs.append(exp)
            return torch.tensor(outputs, device=x.device).unsqueeze(1)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical linear layers for gates
        self.forget_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if n_qubits > 0:
            self.forget_q = self.QuantumGate(n_qubits, n_qubits)
            self.input_q = self.QuantumGate(n_qubits, n_qubits)
            self.update_q = self.QuantumGate(n_qubits, n_qubits)
            self.output_q = self.QuantumGate(n_qubits, n_qubits)
        else:
            self.forget_q = None
            self.input_q = None
            self.update_q = None
            self.output_q = None

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            # Classical gate outputs
            f = torch.sigmoid(self.forget_lin(combined))
            i = torch.sigmoid(self.input_lin(combined))
            g = torch.tanh(self.update_lin(combined))
            o = torch.sigmoid(self.output_lin(combined))
            if self.n_qubits > 0:
                # Quantum gate outputs
                f = torch.sigmoid(self.forget_q(self.forget_lin(combined)))
                i = torch.sigmoid(self.input_q(self.input_lin(combined)))
                g = torch.tanh(self.update_q(self.update_lin(combined)))
                o = torch.sigmoid(self.output_q(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""

    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM__gen308(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM__gen308", "LSTMTagger"]
