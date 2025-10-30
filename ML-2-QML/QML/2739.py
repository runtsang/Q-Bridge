"""Unified quantum regression model with a hybrid confidence head.

This module combines the quantum regression seed with the hybrid quantum
circuit from the binary‑classification seed.  It uses a parameterised
quantum encoder, a variational layer, and a measurement head for the
regression output.  A separate hybrid quantum head computes a confidence
score by evaluating a qiskit circuit on the classical input.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import qiskit
from qiskit import assemble, transpile
from torch.utils.data import Dataset
from typing import Tuple

def generate_superposition_data(
    num_wires: int,
    samples: int,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int, seed: int | None = None):
        self.states, self.labels = generate_superposition_data(num_wires, samples, seed)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumExpectationCircuit:
    """Parameterised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("aer_simulator")
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
            probabilities = counts / self.shots
            return np.sum(states * probabilities)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridExpectationFunction(torch.autograd.Function):
    """Differentiable wrapper around QuantumExpectationCircuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumExpectationCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy()
        expectation = ctx.circuit.run(thetas)
        result = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        thetas = inputs.detach().cpu().numpy()
        grads = []
        for theta in thetas:
            right = ctx.circuit.run(np.array([theta + shift]))
            left = ctx.circuit.run(np.array([theta - shift]))
            grads.append((right - left) / (2 * shift))
        grads = np.array(grads)
        grad_inputs = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grad_inputs * grad_output, None, None

class HybridConfidenceHead(nn.Module):
    """Hybrid quantum head that outputs a confidence score."""
    def __init__(self, n_qubits: int, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumExpectationCircuit(n_qubits)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridExpectationFunction.apply(inputs, self.circuit, self.shift)

class UnifiedQuantumRegression(tq.QuantumModule):
    """Hybrid quantum‑classical regression model with confidence output."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        self.conf_head = HybridConfidenceHead(num_wires)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = states.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=states.device)
        self.encoder(qdev, states)
        self.q_layer(qdev)
        meas = self.measure(qdev)
        pred = self.head(meas).squeeze(-1)
        # Map classical states to a single angle per sample for the hybrid head
        angles = torch.sum(states, dim=1).real
        conf = self.conf_head(angles)
        return pred, conf

__all__ = ["UnifiedQuantumRegression", "RegressionDataset", "generate_superposition_data"]
