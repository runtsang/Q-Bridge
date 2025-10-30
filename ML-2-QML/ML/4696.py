import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import qiskit
import numpy as np
from qiskit import assemble, transpile
from qiskit.circuit import Parameter

class QuantumCircuit:
    """Parametrised quantum circuit executed on Aer simulator."""
    def __init__(self, n_qubits: int, backend, shots: int):
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
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Autograd interface that forwards a tensor through a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift
        gradients = []
        for idx, value in enumerate(input_values):
            exp_r = ctx.quantum_circuit.run([value + shift[idx]])
            exp_l = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(exp_r - exp_l)
        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

class Hybrid(nn.Module):
    """Hybrid layer that applies a quantum expectation to the input."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum convolution filter that applies a random two‑qubit kernel to 2×2 patches."""
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

class HybridQuanvolutionNet(nn.Module):
    """
    Hybrid classical‑quantum network that combines a quantum convolution
    filter with a quantum expectation head.  It can be used for both
    binary classification and regression.
    """
    def __init__(self, task: str = "classification"):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.fc = nn.Linear(4 * 14 * 14, 1)
        self.task = task
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits=1, backend=backend, shots=200, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.fc(features)
        if self.task == "classification":
            probs = torch.sigmoid(logits)
            return torch.cat((probs, 1 - probs), dim=-1)
        else:
            return logits

__all__ = ["HybridQuanvolutionNet"]
