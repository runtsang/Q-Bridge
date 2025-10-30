import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    """Parameterized two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend: AerSimulator, shots: int) -> None:
        self.circuit = QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = QuantumCircuitWrapper._parameter("theta")
        self.circuit.h(all_qubits)
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()
        self.backend = backend
        self.shots = shots

    @staticmethod
    def _parameter(name: str):
        from qiskit.circuit import Parameter
        return Parameter(name)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        if isinstance(result, list):
            return np.array([self._expectation(item) for item in result])
        return np.array([self._expectation(result)])

    def _expectation(self, count_dict: dict) -> float:
        counts = np.array(list(count_dict.values()))
        states = np.array(list(count_dict.keys())).astype(float)
        probabilities = counts / self.shots
        return np.sum(states * probabilities)

class HybridFunction(torch.autograd.Function):
    """Differentiable interface that forwards a scalar to a quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = circuit.run(inputs.tolist())
        out = torch.tensor(expectation, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.full_like(inputs.cpu().numpy(), ctx.shift)
        grads = []
        for val, grad in zip(inputs.cpu().numpy(), grad_output.cpu().numpy()):
            right = ctx.circuit.run([val + shift])
            left = ctx.circuit.run([val - shift])
            grads.append(right - left)
        grads = torch.tensor(grads, device=grad_output.device, dtype=grad_output.dtype)
        return grads * grad_output, None, None

class HybridLayer(nn.Module):
    """Runs a scalar through a quantum circuit to obtain a differentiable expectation."""
    def __init__(self, n_qubits: int, backend: AerSimulator, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat = inputs.view(-1)
        return HybridFunction.apply(flat, self.circuit, self.shift).view(inputs.shape)

class QGateLayer(nn.Module):
    """A small variational circuit that acts as a gate in an LSTM cell."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = nn.Linear(n_wires, n_wires)
        self.rxs = nn.ModuleList([nn.Parameter(torch.randn(1)) for _ in range(n_wires)])
        self.measure = nn.Linear(n_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.encoder(x)
        output = torch.tanh((self.rxs[0] * angles).sum(dim=1, keepdim=True))
        return self.measure(output)

class QuantumQLSTMCell(nn.Module):
    """Combines classical linear gates and quantum gate layer for each LSTM gate."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.forget = QGateLayer(n_qubits)
        self.input = QGateLayer(n_qubits)
        self.update = QGateLayer(n_qubits)
        self.output = QGateLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch, self.linear_forget.out_features, device=device),
            torch.zeros(batch, self.linear_forget.out_features, device=device),
        )

class QuantumHead(nn.Module):
    """Quantum head that uses a hybrid layer to produce a probability."""
    def __init__(self, in_features: int, n_qubits: int, backend: AerSimulator, shots: int, shift: float) -> None:
        super().__init__()
        self.hybrid = HybridLayer(n_qubits, backend, shots, shift)
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x).view(-1)
        return self.hybrid(logits)

class UnifiedQLSTMClassifier(nn.Module):
    """Sequence tagger that can switch between classical and quantum LSTM,
    and ends with a quantum hybrid head."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, backend: AerSimulator = None, shots: int = 100, shift: float = np.pi/2):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QuantumQLSTMCell(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        if n_qubits > 0:
            self.head = QuantumHead(tagset_size, n_qubits, backend, shots, shift)
        else:
            self.head = ClassicalHead(tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        probs = self.head(tag_logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["UnifiedQLSTMClassifier", "QuantumHead", "QuantumQLSTMCell", "QGateLayer", "HybridLayer", "HybridFunction", "QuantumCircuitWrapper"]
