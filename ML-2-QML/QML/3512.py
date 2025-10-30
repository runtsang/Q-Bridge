"""QuanvolutionHybridNet – quantum‑enabled implementation.

This module extends the classical architecture with two quantum
components:
1. **QuantumQuanvolutionFilter** – a variational 2‑qubit kernel that
   acts on 2×2 image patches.  It uses a random layer followed by
   a measurement of Pauli‑Z, mirroring the original
   quanvolution example.
2. **QuantumHybridHead** – a parameterised two‑qubit circuit that
   produces a single expectation value.  It is wrapped in a
   differentiable PyTorch function so that gradients can back‑prop
   through the simulator.
3. **QuanvolutionHybridNet** – combines the filter and head into a
   single quantum‑classical network.  The class inherits from
   ``tq.QuantumModule`` so that the device and batch‑size
   information is propagated automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import qiskit
from qiskit import assemble, transpile

import torchquantum as tq

__all__ = ["QuantumQuanvolutionFilter",
           "QuantumHybridHead",
           "QuanvolutionHybridNet"]


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """
    Quantum kernel applied to each 2×2 patch of the input image.
    The kernel consists of a random circuit followed by a
    measurement of the Pauli‑Z operator on all qubits.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_qubits, bsz=bsz, device=device)
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


class QuantumCircuit:
    """
    Parametrised two‑qubit circuit executed on a Qiskit Aer
    simulator.  The circuit applies a Hadamard, a Ry rotation
    parameterised by ``theta``, and then measures all qubits.
    """
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, qubits)
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
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge between PyTorch and the Qiskit circuit.
    The forward pass evaluates the circuit for the given angles
    and returns the expectation value.  The backward pass
    approximates the gradient via finite‑difference (parameter‑shift).
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit,
                shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expect = ctx.circuit.run(inputs.tolist())
        out = torch.tensor(expect, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        gradients = []
        for inp, s in zip(inputs.tolist(), shift):
            right = ctx.circuit.run([inp + s])
            left = ctx.circuit.run([inp - s])
            gradients.append(right - left)
        grad = torch.tensor(gradients, device=grad_output.device)
        return grad * grad_output, None, None


class QuantumHybridHead(nn.Module):
    """
    Hybrid layer that forwards activations through a Qiskit circuit.
    It accepts a batch of scalars and returns a probability
    via a sigmoid applied to the expectation value.
    """
    def __init__(self, n_qubits: int = 2,
                 backend=None, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        if backend is None:
            backend = qiskit.Aer.get_backend("aer_simulator")
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(HybridFunction.apply(x.squeeze(), self.circuit, self.shift))


class QuanvolutionHybridNet(tq.QuantumModule):
    """
    End‑to‑end hybrid network that combines a quantum quanvolution
    filter with a quantum hybrid head.  It can be used as a drop‑in
    replacement for the classical version or as a fully quantum
    baseline.  The class inherits from ``tq.QuantumModule`` so that
    the quantum device is automatically initialised.
    """
    def __init__(self,
                 filter_out_channels: int = 8,
                 head: nn.Module | None = None) -> None:
        super().__init__()
        self.filter = QuantumQuanvolutionFilter()
        dummy = torch.zeros(1, 1, 28, 28)
        dummy_feat = self.filter(dummy)
        in_features = dummy_feat.size(1)
        self.head = head if head is not None else QuantumHybridHead()
        # Register the head as a sub‑module so that parameters are tracked
        self.add_module("head", self.head)

    def replace_head(self, new_head: nn.Module) -> None:
        """
        Swap the current head for a new one.  Useful for experimenting
        with a purely classical head or a different quantum circuit.
        """
        self.head = new_head
        self.add_module("head", new_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        probs = self.head(features)
        return probs
