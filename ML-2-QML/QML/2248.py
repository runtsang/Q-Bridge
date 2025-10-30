"""Quantum kernel and autoencoder utilities using TorchQuantum and Qiskit."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel implemented with a fixed TorchQuantum ansatz."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.QuantumModule(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.ansatz.ansatz:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.ansatz.ansatz):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b`` using the quantum kernel."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

def quantum_autoencoder(num_latent: int = 3, num_trash: int = 2) -> QuantumCircuit:
    """Build a quantum autoencoder circuit using a swap test."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.append(ansatz, list(range(num_latent + num_trash)))

    qc.barrier()

    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

def sample_autoencoder_output(circuit: QuantumCircuit, shots: int = 1024) -> np.ndarray:
    """Sample measurement outcomes from the autoencoder circuit."""
    sampler = StatevectorSampler()
    result = sampler.run(circuit, shots=shots)
    return result.data

__all__ = [
    "QuantumKernel",
    "kernel_matrix",
    "quantum_autoencoder",
    "sample_autoencoder_output",
]
