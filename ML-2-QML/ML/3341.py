import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit

# ----------------------------------------------------------------------
# Quantum fully‑connected layer (classical wrapper)
# ----------------------------------------------------------------------
class FCL:
    """Parameterised quantum circuit that acts as a linear layer."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 100):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self._build_circuit()

    def _build_circuit(self):
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(self.n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(self.n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Run the circuit for a batch of parameters."""
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: float(t)} for t in thetas],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

# ----------------------------------------------------------------------
# Hybrid classical LSTM
# ----------------------------------------------------------------------
class HybridQLSTM(nn.Module):
    """
    LSTM cell where each gate is computed by a quantum fully connected
    layer (FCL). The circuit outputs a single expectation value that
    is passed through a sigmoid or tanh to form the gate.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_size = input_dim + hidden_dim

        # Quantum linear layers for each gate
        self.forget_gate = FCL(gate_size, shots=200)
        self.input_gate = FCL(gate_size, shots=200)
        self.update_gate = FCL(gate_size, shots=200)
        self.output_gate = FCL(gate_size, shots=200)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch, self.hidden_dim, device=device),
            torch.zeros(batch, self.hidden_dim, device=device),
        )

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)  # (1, gate_size)
            # Convert to numpy for quantum evaluation
            thetas = combined.detach().cpu().numpy().flatten()
            f = torch.sigmoid(torch.tensor(self.forget_gate.run(thetas), device=combined.device))
            i = torch.sigmoid(torch.tensor(self.input_gate.run(thetas), device=combined.device))
            g = torch.tanh(torch.tensor(self.update_gate.run(thetas), device=combined.device))
            o = torch.sigmoid(torch.tensor(self.output_gate.run(thetas), device=combined.device))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

class HybridLSTMTagger(nn.Module):
    """Tagging model that can use either the hybrid classical LSTM
    or a standard nn.LSTM."""
    def __init__(self, embed_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        if n_qubits > 0:
            self.lstm = HybridQLSTM(embed_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(logits, dim=1)

__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
