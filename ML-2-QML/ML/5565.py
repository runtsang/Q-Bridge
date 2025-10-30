import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchquantum as tq
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.quantum_info import SparsePauliOp

class QuantumCircuitWrapper:
    def __init__(self, n_qubits, backend, shots=1024):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self.circuit.ry(self.theta[i], i)
        self.circuit.measure_all()

    def run(self, thetas):
        compiled = transpile(self.circuit, self.backend)
        param_bind = {self.theta[i]: thetas[i] for i in range(len(thetas))}
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, circuit, shift):
        ctx.shift = shift
        ctx.circuit = circuit
        expectations = []
        for sample in inputs:
            exp = circuit.run(sample.tolist())
            expectations.append(exp[0])
        result = torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for sample in inputs:
            g = []
            for val in sample:
                right = ctx.circuit.run([val + shift])
                left = ctx.circuit.run([val - shift])
                g.append(right - left)
            grads.append(g)
        grads = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device)
        return grads * grad_output.unsqueeze(1), None, None

class Hybrid(nn.Module):
    def __init__(self, n_qubits, backend, shots, shift):
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs):
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)

class UnifiedQuanvolutionHybrid(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 10,
                 patch_size: int = 2,
                 n_qubits: int = 4,
                 shift: float = math.pi / 2):
        super().__init__()
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.shift = shift

        # Classical backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Quantum patch filter
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Compute feature map size after backbone
        self.feature_map_size = 28 // 4  # MNIST 28x28 -> 7x7
        self.num_patches = (self.feature_map_size // patch_size) ** 2

        # Classical branch head
        self.classical_head = nn.Sequential(
            nn.Linear(16 * self.feature_map_size * self.feature_map_size, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        # Quantum branch head
        self.quantum_head = nn.Linear(self.num_patches * n_qubits, 4)

        # Hybrid quantum head
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits, backend, shots=100, shift=shift)

        # Final logits head
        self.logits_head = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Classical backbone
        features = self.backbone(x)

        # Quantum patch filtering
        patches = []
        for r in range(0, features.shape[2], self.patch_size):
            for c in range(0, features.shape[3], self.patch_size):
                patch = features[:, r, c, :].view(bsz, 1, 1, -1)
                data = patch.view(bsz, self.n_qubits)
                qdev = tq.QuantumDevice(self.n_qubits, bsz=bsz, device=x.device)
                self.encoder(qdev, data)
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_qubits))
        quantum_features = torch.cat(patches, dim=1)

        # Classical head
        flat = features.view(bsz, -1)
        classical_out = self.classical_head(flat)

        # Quantum head
        quantum_out = self.quantum_head(quantum_features)

        # Combine
        combined = classical_out + quantum_out

        # Hybrid quantum expectation
        expectation = self.hybrid(combined)

        # Logits
        logits = self.logits_head(expectation.unsqueeze(1))
        logits = F.log_softmax(logits, dim=-1)
        return logits

__all__ = ["UnifiedQuanvolutionHybrid"]
