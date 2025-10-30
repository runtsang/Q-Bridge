"""Hybrid classical binary classifier with optional quantum head.

The module defines a reusable `HybridQuantumBinaryClassifier` that can
* run a standard CNN followed by a dense sigmoid head (classical mode)
* or replace the dense head with a variational two‑qubit circuit (quantum mode)
* or switch to a Pennylane‑style hybrid layer for higher‑depth circuits.

The design is inspired by the reference seeds but integrates
all core ideas: dense‑sigmoid, quantum‑expectation, and Q‑LSTM
for sequence tagging.  The public API is kept minimal so that
the class can be dropped into existing training pipelines.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------- #
#  Classical dense head
# --------------------------------------------------------------------------- #
class _DenseSigmoidHead(nn.Module):
    """A small dense layer followed by a sigmoid."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)

# --------------------------------------------------------------------------- #
#  Quantum circuit wrapper (two‑qubit variational)
# --------------------------------------------------------------------------- #
class _TwoQubitVariationalCircuit:
    """Variational circuit with two qubits and a expectation value
    on the number‑of‑bits‑counting (Pauli‑Z) basis.  This is a
    reusable wrapper that can be executed on Aer or Pennylane.
    """
    def __init__(self, n_qubits: int = 2, backend: str = 'aer', shots: int = 100):
        if backend == 'aer':
            import qiskit
            from qiskit import assemble, transpile
            self.backend = qiskit.Aer.get_backend('aer_simulator')
            self.circuit = qiskit.QuantumCircuit(n_qubits)
            self.theta = qiskit.circuit.Parameter('theta')
            # simple entangling set‑up
            self.circuit.h(0)
            self.circuit.cx(0, 1)
            self.circuit.ry(self.theta, 0)
            self.circuit.ry(self.theta, 1)
            self.circuit.measure_all()
            self._run = self._run_aer
        else:
            import pennylane as qml
            self.dev = qml.device('default.qubit', wires=2)
            @qml.qnode(self.dev, interface='torch')
            def circuit(theta):
                qml.Hadamard(0)
                qml.CNOT(0,1)
                qml.RY(theta, 0)
                qml.RY(theta, 1)
                return qml.expval(qml.PauliZ(0))
            self.circuit = circuit
            self._run = self._run_pennylane
        self.shots = shots

    def _run_aer(self, theta: float):
        import qiskit
        from qiskit import assemble, transpile
        bound = self.circuit.bind_parameters({self.theta: theta})
        compiled = transpile(bound, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        probs = {k: v / self.shots for k, v in counts.items()}
        # expectation of Z on first qubit
        exp = sum(((-1)**int(bit[0])) * p for bit, p in probs.items())
        return exp

    def _run_pennylane(self, theta: float):
        return self.circuit(theta)

    def run(self, theta: float):
        return self._run(theta)

# --------------------------------------------------------------------------- #
#  Differentiable quantum expectation
# --------------------------------------------------------------------------- #
class _QuantumExpectationLayer(nn.Module):
    """A torch autograd Function that forwards a scalar through a variational
    circuit and back‑propagates via finite‑difference central
    difference.  The implementation is intentionally lightweight
    and uses only a single parameter per forward pass, which is
    sufficient for the binary classification task.
    """
    class _Func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, theta: torch.Tensor, circuit: _TwoQubitVariationalCircuit, shift: float):
            ctx.circuit = circuit
            ctx.shift = shift
            theta_np = theta.detach().cpu().numpy().item()
            exp = circuit.run(theta_np)
            result = torch.tensor([exp], device=theta.device, dtype=theta.dtype)
            ctx.save_for_backward(theta, result)
            return result

        @staticmethod
        def backward(ctx, grad_output):
            theta, _ = ctx.saved_tensors
            shift = ctx.shift
            theta_np = theta.detach().cpu().numpy().item()
            exp_plus = ctx.circuit.run(theta_np + shift)
            exp_minus = ctx.circuit.run(theta_np - shift)
            grad = (exp_plus - exp_minus) / (2 * shift)
            grad_tensor = torch.tensor([grad], device=theta.device, dtype=theta.dtype)
            return grad_tensor * grad_output, None, None

    def __init__(self, circuit: _TwoQubitVariationalCircuit, shift: float = np.pi/2):
        super().__init__()
        self.circuit = circuit
        self.shift = shift

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self._Func.apply(theta, self.circuit, self.shift)

# --------------------------------------------------------------------------- #
#  Hybrid binary classifier
# --------------------------------------------------------------------------- #
class HybridQuantumBinaryClassifier(nn.Module):
    """CNN backbone followed by either a classical dense head or a quantum
    variational head.  The `use_quantum` flag controls which head is used.
    """
    def __init__(self,
                 use_quantum: bool = False,
                 quantum_backend: str = 'aer',
                 quantum_shots: int = 100,
                 shift: float = np.pi/2):
        super().__init__()
        # CNN backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Head
        self.use_quantum = use_quantum
        if use_quantum:
            self.circuit = _TwoQubitVariationalCircuit(backend=quantum_backend,
                                                       shots=quantum_shots)
            self.head = _QuantumExpectationLayer(self.circuit, shift=shift)
        else:
            self.head = _DenseSigmoidHead(in_features=1, shift=shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.use_quantum:
            logits = x.squeeze(-1)
            probs = self.head(logits)
        else:
            probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

# --------------------------------------------------------------------------- #
#  Quantum LSTM for sequence tagging
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits."""
    class _QLayer(nn.Module):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Simple parameterized Rx gates per wire
            self.rxs = nn.ModuleList([nn.Parameter(torch.randn(1)) for _ in range(n_wires)])
            self.measure = nn.Linear(n_wires, 1)  # placeholder for measurement

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Treat x as parameters for Rx gates
            out = torch.sin(x)  # mock quantum evolution
            out = self.measure(out)
            return out

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self._QLayer(n_qubits)
        self.input = self._QLayer(n_qubits)
        self.update = self._QLayer(n_qubits)
        self.output = self._QLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple = None):
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
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0):
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

__all__ = ["HybridQuantumBinaryClassifier", "QLSTM", "LSTMTagger"]
