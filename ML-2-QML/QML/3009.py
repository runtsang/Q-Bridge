import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit as QC, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    """Parametrised multi‑qubit circuit that returns the Z‑expectation for each qubit."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or AerSimulator()
        self.circuit = QC(n_qubits)
        self.theta = [QC.Parameter(f'theta_{i}') for i in range(n_qubits)]
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        for i, th in enumerate(self.theta):
            self.circuit.ry(th, i)
        self.circuit.measure_all()

    def _expectation_z(self, counts):
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        expectation = np.zeros(self.n_qubits)
        for i in range(self.n_qubits):
            bits = ((states >> i) & 1).astype(int)
            expectation[i] = np.sum((1 - 2 * bits) * probs)
        return expectation

    def run(self, angles):
        """angles: numpy array of shape (batch, n_qubits)"""
        results = []
        compiled = transpile(self.circuit, self.backend)
        for ang in angles:
            param_binds = [{th: ang[i].item()} for i, th in enumerate(self.theta)]
            qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
            job = self.backend.run(qobj)
            counts = job.result().get_counts()
            results.append(self._expectation_z(counts))
        return np.array(results)

class QuantumExpectationFunction(autograd.Function):
    """Differentiable interface that runs the quantum circuit with the parameter‑shift rule."""
    @staticmethod
    def forward(ctx, angles: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        angles_np = angles.detach().cpu().numpy()
        expectation = ctx.circuit.run(angles_np)
        result = torch.tensor(expectation, device=angles.device, dtype=angles.dtype)
        ctx.save_for_backward(angles, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        angles, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for b in range(angles.shape[0]):
            for q in range(angles.shape[1]):
                ang_plus = angles.clone()
                ang_minus = angles.clone()
                ang_plus[b, q] += shift
                ang_minus[b, q] -= shift
                exp_plus = ctx.circuit.run(ang_plus.detach().cpu().numpy())
                exp_minus = ctx.circuit.run(ang_minus.detach().cpu().numpy())
                grads.append((exp_plus[b, q] - exp_minus[b, q]) / 2)
        grads = torch.tensor(grads, device=angles.device, dtype=angles.dtype)
        grads = grads.view_as(angles)
        return grads * grad_output, None, None

class QuantumGate(nn.Module):
    """Linear layer that feeds angles into a quantum circuit and returns the expectation."""
    def __init__(self, in_features: int, n_qubits: int, backend=None, shots: int = 1024, shift: float = np.pi/2):
        super().__init__()
        self.linear = nn.Linear(in_features, n_qubits)
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.linear(x)
        return QuantumExpectationFunction.apply(angles, self.circuit, self.shift)

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM where each gate is a QuantumGate."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int,
                 backend=None, shots: int = 1024, shift: float = np.pi/2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_gate = QuantumGate(input_dim + hidden_dim, n_qubits, backend, shots, shift)
        self.input_gate  = QuantumGate(input_dim + hidden_dim, n_qubits, backend, shots, shift)
        self.update_gate = QuantumGate(input_dim + hidden_dim, n_qubits, backend, shots, shift)
        self.output_gate = QuantumGate(input_dim + hidden_dim, n_qubits, backend, shots, shift)

        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin  = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _init_states(self, inputs: torch.Tensor, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor, states=None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = self.forget_gate(self.forget_lin(combined))
            i = self.input_gate(self.input_lin(combined))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = self.output_gate(self.output_lin(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

class LSTMTagger(nn.Module):
    """Sequence tagging model that can use the quantum LSTM or a classical LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, backend=None,
                 shots: int = 1024, shift: float = np.pi/2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits,
                              backend, shots, shift)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
