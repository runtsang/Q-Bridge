import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
import numpy as np

# ------------------------------------------------------------------
# Quantum circuit for the hybrid head
# ------------------------------------------------------------------
class QuantumCircuit:
    """Parameterized two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = qiskit.transpile(self._circuit, self.backend)
        qobj = qiskit.assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

# ------------------------------------------------------------------
# Hybrid autograd function that bridges PyTorch and the quantum circuit
# ------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.detach().cpu().numpy().flatten())
        return torch.tensor(expectation, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shift = np.ones_like(ctx.shift) * ctx.shift
        grads = []
        for val in ctx.circuit.run(np.array([ctx.shift])):
            grads.append(val)
        grads = torch.tensor(grads, dtype=torch.float32, device=grad_output.device)
        return grads * grad_output, None, None

# ------------------------------------------------------------------
# Hybrid head that forwards through the quantum circuit
# ------------------------------------------------------------------
class Hybrid(nn.Module):
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

# ------------------------------------------------------------------
# Quantum “gate” implemented with torchquantum
# ------------------------------------------------------------------
class QLayer(tq.QuantumModule):
    """
    Small quantum circuit that acts as a gate in the LSTM cell.
    The circuit consists of parameterized RX rotations followed by
    a chain of CNOTs and measurement.
    """
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)

# ------------------------------------------------------------------
# Quantum‑enhanced LSTM with a hybrid head
# ------------------------------------------------------------------
class QLSTMHybrid(nn.Module):
    """
    Drop‑in replacement for the original QLSTM that uses quantum
    gates for the internal LSTM cell and a quantum expectation head.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates for each LSTM gate
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        # Linear maps from classical concatenated state to qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Final fully‑connected layer
        self.fc = nn.Linear(hidden_dim, 1)

        # Hybrid head that runs a quantum circuit
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits, backend, shots=200, shift=np.pi / 2)

    def _init_states(self, batch_size: int, device: torch.device):
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, features).

        Returns
        -------
        torch.Tensor
            Binary logits of shape (batch, seq_len, 2).
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        hx, cx = self._init_states(batch_size, device)
        outputs = []

        for t in range(seq_len):
            xt = x[:, t, :]  # (batch, 1)
            combined = torch.cat([xt, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))

        out = torch.cat(outputs, dim=1)            # (batch, seq_len, hidden)
        logits = self.fc(out)                      # (batch, seq_len, 1)
        logits = self.hybrid(logits)               # (batch, seq_len, 1)
        return torch.cat((logits, 1 - logits), dim=-1)

__all__ = ["QLSTMHybrid"]
