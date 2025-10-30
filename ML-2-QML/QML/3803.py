"""Quantum version of QuanvolutionHybrid."""

import numpy as np
import torch
import torch.nn as nn
import torch.quantum as tq
import qiskit
from qiskit import assemble, transpile


class QuanvolutionFilter(tq.QuantumModule):
    """Random two‑qubit quantum kernel applied to 2×2 image patches."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
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
                    [x[:, r, c], x[:, r, c + 1],
                     x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuantumCircuit:
    """Parametrised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(list(range(n_qubits)))
        self._circuit.barrier()
        self._circuit.ry(self.theta, list(range(n_qubits)))
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
    """Differentiable wrapper between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = circuit.run(inputs.tolist())
        out = torch.tensor([expectation])
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        grads = []
        for val in inputs.tolist():
            pos = ctx.circuit.run([val + shift])
            neg = ctx.circuit.run([val - shift])
            grads.append(pos - neg)
        grads = torch.tensor([grads]).float()
        return grads * grad_output.float(), None, None


class Hybrid(nn.Module):
    """Quantum expectation head."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.circuit, self.shift)


class QuanvolutionHybrid(nn.Module):
    """Quantum quanvolution filter followed by a quantum expectation head."""
    def __init__(self, in_channels: int = 1, n_qubits: int = 4) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter(n_wires=n_qubits)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.head = Hybrid(n_qubits, backend, shots=100, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        probs = self.head(features)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuanvolutionFilter", "QuantumCircuit", "HybridFunction",
           "Hybrid", "QuanvolutionHybrid"]
