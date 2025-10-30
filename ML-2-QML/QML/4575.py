"""Quantum kernel and sampler implementations using Qiskit and TorchQuantum.
The module exposes two helper constructors that return ready‑to‑use objects:
* quantum_sampler_qnn(num_qubits=4, reps=3)  → Qiskit SamplerQNN
* quantum_kernel() → TorchQuantum Kernel module
"""
from __future__ import annotations

import numpy as np
from typing import Sequence

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit_machine_learning.circuit.library import RawFeatureVector

def quantum_sampler_qnn(num_qubits: int = 4, reps: int = 3) -> QiskitSamplerQNN:
    """Return a Qiskit SamplerQNN that maps an input vector to a probability
    distribution over 2^num_qubits outcomes.  The circuit uses a RealAmplitudes
    ansatz followed by a swap‑test for measurement."""
    from qiskit.circuit import ParameterVector

    input_params = ParameterVector("input", num_qubits)
    weight_params = ParameterVector("weight", num_qubits * reps * 2)

    qr = QuantumRegister(num_qubits + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    circuit.compose(RealAmplitudes(num_qubits, reps=reps), range(num_qubits), inplace=True)
    circuit.barrier()

    anc = num_qubits
    circuit.h(anc)
    for i in range(num_qubits):
        circuit.cswap(anc, i, anc)
    circuit.h(anc)
    circuit.measure(anc, cr[0])

    sampler = StatevectorSampler()
    return QiskitSamplerQNN(
        circuit=circuit,
        input_params=input_params,
        weight_params=weight_params,
        sampler=sampler,
    )

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a list of programmable quantum gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = [
    "quantum_sampler_qnn",
    "QuantumKernel",
    "kernel_matrix",
]
