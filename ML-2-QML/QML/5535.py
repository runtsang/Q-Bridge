import numpy as np
import torch
from torch import nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
import qiskit
from qiskit import assemble, transpile

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel using TorchQuantum."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.q_device.reset_states(x.shape[0])
        self.encoder(self.q_device, x)
        self.random_layer(self.q_device)
        self.measure(self.q_device)
        states_x = self.q_device.states.view(-1)[0]
        self.q_device.reset_states(y.shape[0])
        self.encoder(self.q_device, -y)
        self.random_layer(self.q_device)
        self.measure(self.q_device)
        states_y = self.q_device.states.view(-1)[0]
        return torch.abs(states_x * states_y)

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum filter applying a two‑qubit kernel to 2x2 patches."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
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
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuantumFCL(tq.QuantumModule):
    """Parameterized quantum circuit acting as a fully connected layer (Qiskit backend)."""
    def __init__(self, n_qubits: int = 1, shots: int = 100):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()

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
        return np.array([expectation(result)])

class HybridQuantumBinaryClassifier(nn.Module):
    """Hybrid network: fully connected layers followed by quantum expectation head."""
    def __init__(self, n_qubits: int = 1, shift: float = np.pi / 2):
        super().__init__()
        self.fc1 = nn.Linear(784, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.quantum_circuit = QuantumFCL(n_qubits, shots=100)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(inputs))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        expectation = torch.tensor(self.quantum_circuit.run(x.detach().cpu().numpy()))
        probs = torch.sigmoid(expectation)
        return torch.cat((probs, 1 - probs), dim=-1)

class QuantumKernelHybrid(nn.Module):
    """Composable hybrid model combining quantum kernel, filter, fully‑connected layer and classifier."""
    def __init__(self):
        super().__init__()
        self.kernel = QuantumKernel()
        self.filter = QuantumQuanvolutionFilter()
        self.fcl = QuantumFCL()
        self.classifier = HybridQuantumBinaryClassifier()

    def forward(self, data: torch.Tensor, params: np.ndarray) -> torch.Tensor:
        features = self.filter(data)
        fcl_output = self.fcl.run(features.detach().cpu().numpy())
        logits = self.classifier(torch.tensor(fcl_output, dtype=torch.float32))
        kernel_val = self.kernel(features, features)
        return logits, kernel_val

__all__ = ["QuantumKernel", "QuantumQuanvolutionFilter", "QuantumFCL",
           "HybridQuantumBinaryClassifier", "QuantumKernelHybrid"]
