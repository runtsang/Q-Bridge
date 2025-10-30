import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import qiskit
from qiskit import assemble, transpile

class QuantumCircuit:
    """Parametrised two‑qubit circuit used by the hybrid layer."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def exp(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([exp(r) for r in result])
        return np.array([exp(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface that forwards inputs through a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        exp_val = ctx.circuit.run(inputs.cpu().numpy())
        out = torch.tensor(exp_val, dtype=torch.float, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_vals = inputs.cpu().numpy()
        shift_vals = np.full_like(input_vals, ctx.shift)
        grads = []
        for v, s in zip(input_vals, shift_vals):
            exp_r = ctx.circuit.run([v + s])[0]
            exp_l = ctx.circuit.run([v - s])[0]
            grads.append(exp_r - exp_l)
        grad_inputs = torch.tensor(grads, dtype=torch.float, device=inputs.device)
        return grad_inputs * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum‑enhanced activation layer."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor):
        return HybridFunction.apply(inputs.squeeze(), self.circuit, self.shift)

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Random two‑qubit quantum kernel applied to 2x2 image patches."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QLSTM(tq.QuantumModule):
    """LSTM cell where gates are realised by small quantum circuits."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)
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
        batch = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch, self.hidden_dim, device=device),
            torch.zeros(batch, self.hidden_dim, device=device),
        )

class QuanvolutionHybridNet(tq.QuantumModule):
    """Hybrid network that combines a quantum quanvolution filter,
    a quantum LSTM for sequential modelling, and a quantum‑hybrid head."""
    def __init__(self, input_channels: int = 3, hidden_dim: int = 64, n_qubits: int = 4):
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.lstm = QLSTM(input_dim=4 * 14 * 14, hidden_dim=hidden_dim, n_qubits=n_qubits)
        self.fc1 = nn.Linear(hidden_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(self.fc3.out_features, backend, shots=100, shift=np.pi / 2)

    def forward(self, x: torch.Tensor):
        features = self.qfilter(x)          # (batch, 4*14*14)
        seq = features.unsqueeze(1)         # (batch, 1, 4*14*14)
        lstm_out, _ = self.lstm(seq)
        hidden = lstm_out.squeeze(1)        # (batch, hidden_dim)
        x = F.relu(self.fc1(hidden))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = self.hybrid(logits).squeeze()
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["QuanvolutionHybridNet"]
