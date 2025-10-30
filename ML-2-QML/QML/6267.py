import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
import torchquantum as tq

class QuantumPatchEncoder(tq.QuantumModule):
    """Parameter‑free encoder that maps a 2×2 patch to a 4‑qubit circuit."""
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
                patches.append(measurement)
        return torch.cat(patches, dim=1)

class QuanvolutionFilter(nn.Module):
    """Quantum filter that applies the patch encoder to every 2×2 patch."""
    def __init__(self, backend, shots: int) -> None:
        super().__init__()
        self.encoder = QuantumPatchEncoder()
        self.backend = backend
        self.shots = shots

    def run(self, patch: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(4, bsz=patch.shape[0], device=patch.device)
        return self.encoder(qdev, patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = x.shape
        patches = []
        for r in range(0, h, 2):
            for c in range(0, w, 2):
                patch = x[:, :, r:r+2, c:c+2]
                patch = patch.view(bsz, 4)
                patches.append(self.run(patch))
        return torch.cat(patches, dim=1)

class QuantumCircuit(nn.Module):
    """Two‑qubit variational circuit used as the final head."""
    def __init__(self, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.backend = backend
        self.shots = shots
        self.shift = shift
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit = qiskit.QuantumCircuit(2)
        self.circuit.h(0)
        self.circuit.h(1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.theta, 0)
        self.circuit.ry(self.theta, 1)
        self.circuit.measure_all()

    def run(self, angle: torch.Tensor) -> torch.Tensor:
        comp = transpile(self.circuit, self.backend)
        param_dict = {self.theta: angle.item()}
        qobj = assemble(comp, shots=self.shots, parameter_binds=[param_dict])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        return self._expectation(result)

    @staticmethod
    def _expectation(counts: dict) -> torch.Tensor:
        probs = torch.tensor(list(counts.values())) / sum(counts.values())
        states = torch.tensor([int(k, 2) for k in counts.keys()], dtype=torch.float)
        return torch.sum(states * probs)

class QuantumExpectationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, angle: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.shift = shift
        ctx.angle = angle
        return circuit.run(angle)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        angle = ctx.angle
        shift = ctx.shift
        circuit = ctx.circuit
        exp_plus = circuit.run(angle + shift)
        exp_minus = circuit.run(angle - shift)
        grad = (exp_plus - exp_minus) / (2 * torch.sin(shift))
        return grad_output * grad, None, None

class HybridHead(nn.Module):
    """Quantum head that outputs a single expectation value."""
    def __init__(self, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(backend, shots, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return QuantumExpectationFunction.apply(x, self.circuit, self.circuit.shift)

class HybridNet(nn.Module):
    """Full hybrid network combining quanvolution, classical CNN, and quantum head."""
    def __init__(self) -> None:
        super().__init__()
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.qfilter = QuanvolutionFilter(backend, shots=200)
        self.conv1 = nn.Conv2d(4, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(60, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = HybridHead(backend, shots=200, shift=3.14159 / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, 1, 28, 28)
        features = self.qfilter(x)
        features = features.view(x.size(0), 4, 14, 14)
        x = F.relu(self.conv1(features))
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
        prob = self.head(x).squeeze(-1).unsqueeze(-1)
        return torch.cat((prob, 1 - prob), dim=-1)
