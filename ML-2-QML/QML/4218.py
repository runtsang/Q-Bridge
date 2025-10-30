import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Quantum utilities – parameterised two‑qubit circuit
# --------------------------------------------------------------------------- #
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.fake_provider import FakeProvider

class QuantumCircuit:
    """Two‑qubit parameterised circuit executed on a simulator."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 100) -> None:
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        # Simple variational layer
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of parameters."""
        compiled = transpile(self.circuit, self.backend)
        # Bind each set of parameters to a separate job
        qobjs = []
        for p in params:
            bind = {self.theta: p}
            qobj = assemble(compiled, parameter_binds=[bind], shots=self.shots)
            qobjs.append(qobj)
        jobs = [self.backend.run(qobj) for qobj in qobjs]
        results = [job.result() for job in jobs]
        expectations = []
        for res in results:
            counts = res.get_counts()
            exp = self._expectation(counts)
            expectations.append(exp)
        return np.array(expectations)

    @staticmethod
    def _expectation(counts: dict) -> float:
        total = sum(counts.values())
        exp = sum(int(state, 2) * (cnt / total) for state, cnt in counts.items())
        return exp


# --------------------------------------------------------------------------- #
#  Differentiable bridge between PyTorch and the quantum circuit
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Parameter‑shift rule gradient for the quantum expectation layer."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Convert tensor to numpy array of shape (batch,)
        params = inputs.detach().cpu().numpy()
        expectations = circuit.run(params)
        result = torch.tensor(expectations, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs)
        return result.unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        # Parameter‑shift rule (finite‑difference)
        with torch.no_grad():
            pos = inputs + shift
            neg = inputs - shift
        exp_pos = ctx.circuit.run(pos.detach().cpu().numpy())
        exp_neg = ctx.circuit.run(neg.detach().cpu().numpy())
        grad = (exp_pos - exp_neg) / (2 * shift)
        grad = torch.tensor(grad, device=inputs.device, dtype=inputs.dtype)
        return grad * grad_output.squeeze(-1), None, None


# --------------------------------------------------------------------------- #
#  Quantum‑parameterised head – returns a single expectation value
# --------------------------------------------------------------------------- #
class QuantumHybridHead(nn.Module):
    """Quantum expectation head used as the final decision unit."""
    def __init__(
        self,
        n_qubits: int = 2,
        backend=None,
        shots: int = 100,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expectation value per sample
        return HybridFunction.apply(inputs, self.circuit, self.shift)


# --------------------------------------------------------------------------- #
#  Quantum LSTM – gates realised by small quantum circuits
# --------------------------------------------------------------------------- #
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumLSTMCell(tq.QuantumModule):
    """Quantum gate implementation of an LSTM cell."""
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        # Encoder mapping classical inputs to qubit angles
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
                {"input_idx": [2], "func": "rx", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        # Trainable rotation gates
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        # Entangling pattern
        for wire in range(self.n_qubits - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        tqf.cnot(qdev, wires=[self.n_qubits - 1, 0])
        return self.measure(qdev)


class QuantumLSTM(tq.QuantumModule):
    """Full quantum LSTM with separate gates implemented by QuantumLSTMCell."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        # Linear projections to qubit space
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        # Quantum gate layers
        self.forget_gate = QuantumLSTMCell(n_qubits)
        self.input_gate = QuantumLSTMCell(n_qubits)
        self.update_gate = QuantumLSTMCell(n_qubits)
        self.output_gate = QuantumLSTMCell(n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# --------------------------------------------------------------------------- #
#  Quantum fully‑connected layer – inspired by Quantum‑NAT
# --------------------------------------------------------------------------- #
class QuantumFC(tq.QuantumModule):
    """Quantum layer that replaces a fully‑connected projection."""
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


# --------------------------------------------------------------------------- #
#  Combined hybrid classifier – quantum‑enhanced head
# --------------------------------------------------------------------------- #
class HybridBinaryClassifierQuantum(nn.Module):
    """CNN → (optional LSTM) → Quantum Hybrid Head → Linear → Binary output."""
    def __init__(
        self,
        use_lstm: bool = False,
        lstm_hidden: int = 128,
        n_qubits: int = 2,
        backend=None,
        shots: int = 100,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.cnn = ClassicalCNN()
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = QuantumLSTM(
                input_dim=self.cnn.out_features,
                hidden_dim=lstm_hidden,
                n_qubits=n_qubits,
            )
        else:
            self.lstm = None
        self.quantum_head = QuantumHybridHead(
            n_qubits=n_qubits, backend=backend, shots=shots, shift=shift
        )
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        if self.use_lstm:
            x, _ = self.lstm(x.unsqueeze(1))
            x = x.squeeze(1)
        q_out = self.quantum_head(x)
        logits = self.linear(q_out)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = [
    "QuantumCircuit",
    "HybridFunction",
    "QuantumHybridHead",
    "QuantumLSTM",
    "QuantumFC",
    "HybridBinaryClassifierQuantum",
]
