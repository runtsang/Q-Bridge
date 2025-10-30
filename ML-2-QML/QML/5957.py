import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
import torchquantum as tq

class QuantumCircuit:
    """Parametrised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
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

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = circuit.run(inputs.tolist())
        output = torch.tensor(expectation, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.cpu().numpy()) * ctx.shift
        grads = []
        for val, s in zip(inputs.cpu().numpy(), shift):
            right = ctx.circuit.run([val + s])
            left = ctx.circuit.run([val - s])
            grads.append(right - left)
        grad_inputs = torch.tensor(grads, device=inputs.device, dtype=inputs.dtype)
        return grad_inputs * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum expectation head."""
    def __init__(self, backend, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(), self.circuit, self.shift)

class QuantumFilter(tq.QuantumModule):
    """Patch‑wise quantum kernel inspired by quanvolution."""
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
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class HybridClassifier(nn.Module):
    """Hybrid quantum‑classical binary classifier that unifies the two reference designs."""
    def __init__(self, backend=None, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        self.qfilter = QuantumFilter()
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(self.backend, shots, shift)
        self.fc1 = nn.Linear(4 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        x = self.hybrid(logits)
        probs = torch.sigmoid(x)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridClassifier"]
