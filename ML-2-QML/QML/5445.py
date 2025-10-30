import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

class QuantumCircuit:
    """Parameterized single‑qubit circuit used as a quantum head."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.n_qubits = n_qubits
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
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

class HybridFunctionQuantum(torch.autograd.Function):
    """Differentiable expectation head that uses a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float = 0.0) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = circuit.run(inputs.tolist())
        result = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return torch.sigmoid(result)
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs) * ctx.shift
        gradients = []
        for idx, val in enumerate(inputs):
            right = ctx.circuit.run([val + shift[idx]]).item()
            left  = ctx.circuit.run([val - shift[idx]]).item()
            gradients.append(right - left)
        gradients = torch.tensor(gradients)
        return gradients * grad_output, None, None

class QuantumSelfAttention:
    """Quantum implementation of a self‑attention block."""
    def __init__(self, n_qubits: int, backend):
        self.n_qubits = n_qubits
        self.backend = backend
    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> np.ndarray:
        circuit = qiskit.QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3*i], i)
            circuit.ry(rotation_params[3*i+1], i)
            circuit.rz(rotation_params[3*i+2], i)
        for i in range(self.n_qubits-1):
            circuit.crx(entangle_params[i], i, i+1)
        circuit.measure_all()
        job = qiskit.execute(circuit, self.backend, shots=1024)
        result = job.result().get_counts()
        expectations = []
        for i in range(self.n_qubits):
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            probs = counts / 1024
            expectation = np.sum(states * probs)
            expectations.append(expectation)
        return np.array(expectations)

class QuantumFullyConnectedLayer(nn.Module):
    """Quantum‑inspired fully‑connected layer built from many single‑qubit circuits."""
    def __init__(self, in_features: int, out_features: int, backend, shots: int = 100):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.backend = backend
        self.shots = shots
        self.linear = nn.Linear(in_features, out_features)
        self.circuits = [QuantumCircuit(1, backend, shots) for _ in range(out_features)]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        outputs = []
        for i, circuit in enumerate(self.circuits):
            angles = self.linear(x)[:, i]  # shape: (batch,)
            out = []
            for a in angles:
                res = circuit.run([a.item()])
                out.append(res[0])
            outputs.append(torch.tensor(out))
        return torch.stack(outputs, dim=1)

class QuantumQCNet(nn.Module):
    """Hybrid convolutional network that uses quantum attention and a quantum head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        backend = qiskit.Aer.get_backend("qasm_simulator")

        # Quantum attention on 120 qubits
        self.attention = QuantumSelfAttention(120, backend)

        # Quantum fully‑connected layers
        self.fc1 = QuantumFullyConnectedLayer(120, 84, backend, shots=100)
        self.fc2 = QuantumFullyConnectedLayer(84, 1, backend, shots=100)

        # Quantum expectation head
        self.head_circuit = QuantumCircuit(1, backend, shots=100)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)

        # Prepare rotation and entangle parameters for attention
        rotation = np.random.randn(120 * 3)  # placeholder; in practice learnable
        entangle  = np.random.randn(119)    # placeholder

        x = torch.from_numpy(self.attention.run(rotation, entangle)).float()

        x = self.fc1(x)
        x = self.fc2(x)

        probs = HybridFunctionQuantum.apply(x, self.head_circuit, 0.0)
        return torch.cat((probs, 1 - probs), dim=-1)

class QLSTMQuantum(tq.QuantumModule):
    """Quantum‑enhanced LSTM cell using torchquantum."""
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
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
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

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
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

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

__all__ = ["QuantumCircuit", "HybridFunctionQuantum", "QuantumSelfAttention",
           "QuantumFullyConnectedLayer", "QuantumQCNet", "QLSTMQuantum"]
