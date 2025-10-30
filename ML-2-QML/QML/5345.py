"""Hybrid quantum‑classical network that merges a quanvolution filter with a
parameterised quantum expectation head.  The implementation follows the
original quanvolution example but replaces the deterministic filter with a
quantum kernel.  A Hybrid layer forwards activations through a simple two‑qubit
circuit, and a FastBaseEstimator wrapper is provided for rapid experimentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import qiskit
import numpy as np

from FastBaseEstimator import FastBaseEstimator


class QuantumCircuit:
    """Simple two‑qubit parametrised circuit executed on Aer."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
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
        compiled = qiskit.transpile(self.circuit, self.backend)
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
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and a parametrised quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit

        expectation = ctx.circuit.run(inputs.tolist())
        result = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift

        gradients = []
        for idx, val in enumerate(inputs.tolist()):
            expectation_right = ctx.circuit.run([val + shift[idx]])
            expectation_left = ctx.circuit.run([val - shift[idx]])
            gradients.append(expectation_right - expectation_left)

        gradients = torch.tensor(gradients, dtype=torch.float32)
        return gradients * grad_output, None, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.circuit, self.shift)


class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""

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


class QuanvolutionHybridNet(nn.Module):
    """Quantum‑classical network combining a quanvolution filter with a
    parameterised quantum expectation head.  The architecture mirrors the
    original QCNet but replaces the linear head with a hybrid layer.
    """

    def __init__(self, use_quantum_head: bool = True, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)
        self.use_quantum_head = use_quantum_head
        backend = qiskit.Aer.get_backend("aer_simulator")
        if use_quantum_head:
            self.hybrid = Hybrid(10, backend, shots=100, shift=shift)
        else:
            self.hybrid = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        if self.hybrid:
            logits = self.hybrid(logits)
        return F.log_softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables,
        parameter_sets,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ):
        """Evaluate the quantum network for a collection of input vectors and
        observables.  The estimator leverages the quantum FastBaseEstimator
        to produce shot‑noised expectation values if requested.
        """
        estimator = FastBaseEstimator(self)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)


__all__ = ["QuanvolutionHybridNet", "Hybrid", "HybridFunction", "QuantumCircuit"]
