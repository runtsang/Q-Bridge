"""Quantum hybrid model integrating a parameterized convolution circuit, a quantum kernel, and a quantum fully‑connected layer.

This module implements ConvGen225Quantum, a drop‑in replacement for the classical Conv filter that leverages
Qiskit for the convolution‑style circuit, TorchQuantum for the kernel and fully‑connected
operations, and a simple measurement‑based output.  It can be used in place of the
original Conv() function while providing quantum‑centric experiments.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# Quantum convolution circuit (from Conv.py QML)
class QuantumConvCircuit:
    def __init__(self, n_qubits: int, backend, shots: int, threshold: float):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.threshold = threshold
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Execute the circuit on a single data patch."""
        param_binds = [{self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(data)}]
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

# Quantum kernel ansatz (from QuantumKernelMethod.py QML)
class QuantumKernel(tq.QuantumModule):
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.random_layer = tq.RandomLayer(n_ops=10, wires=list(range(n_wires)))

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        qdev.reset_states(x.shape[0])
        self.random_layer(qdev)
        # encode x
        for i in range(self.n_wires):
            tq.RX(qdev, wires=i, params=x[:, i])
        # encode y with inverse
        for i in range(self.n_wires):
            tq.RX(qdev, wires=i, params=-y[:, i])

# Quantum fully‑connected layer (from QuantumNAT.py QML)
class QuantumFC(tq.QuantumModule):
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        qdev.reset_states(1)
        self.random_layer(qdev)
        self.measure(qdev)

# Hybrid quantum model
class ConvGen225Quantum(tq.QuantumModule):
    """Quantum hybrid model combining a convolution‑style circuit, a quantum kernel, and a quantum fully‑connected head."""
    def __init__(
        self,
        kernel_size: int = 2,
        n_wires: int = 4,
        shots: int = 100,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.n_wires = n_wires
        self.shots = shots
        self.threshold = threshold
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        # Quantum convolution circuit
        self.quantum_conv = QuantumConvCircuit(n_wires, self.backend, shots, threshold)

        # Quantum kernel
        self.kernel = QuantumKernel(n_wires)

        # Quantum fully‑connected layer
        self.q_layer = QuantumFC(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assume x is a batch of flattened images of shape (batch, n_wires)
        batch_vals = [self.quantum_conv.run(p) for p in x.detach().cpu().numpy()]
        conv_tensor = torch.tensor(batch_vals, dtype=torch.float32).unsqueeze(1)

        # Quantum kernel similarity with a learned prototype
        proto = torch.randn(1, self.n_wires, device=x.device)
        self.kernel(self.kernel.q_device, conv_tensor, proto)
        kernel_val = torch.abs(self.kernel.q_device.states.view(-1)[0])

        # Quantum fully‑connected classification
        self.q_layer(self.q_layer.q_device)
        out = self.q_layer.q_device.states.view(-1)[0]

        # Combine outputs
        return out + kernel_val

def Conv() -> ConvGen225Quantum:
    """Return a fully‑initialized quantum hybrid model."""
    return ConvGen225Quantum()
