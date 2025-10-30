"""Hybrid quantum‑classical model combining a parameterized quantum circuit for a
fully connected layer (from the FCL example) with a quanvolutional filter
implemented in TorchQuantum.  The class exposes a ``run`` method that evaluates
the fully connected circuit for a list of parameters, and a ``forward`` method
that applies the quanvolution filter to an input image tensor.

The implementation is intentionally lightweight: the quantum circuit operates
on a single qubit and is executed on the Aer simulator; the quanvolution
filter uses a 4‑wire device and a random two‑qubit kernel.
"""

import numpy as np
import qiskit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from typing import Iterable

class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""

    def __init__(self) -> None:
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

class HybridQuantumLayer(nn.Module):
    """
    Hybrid quantum‑classical neural network.

    Attributes
    ----------
    q_circuit : qiskit.QuantumCircuit
        Parameterized circuit for the fully connected quantum layer.
    qfilter : QuanvolutionFilter
        Quanvolution filter implemented with TorchQuantum.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 100) -> None:
        super().__init__()
        # Quantum circuit for fully connected layer
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(theta, range(n_qubits))
        self._circuit.measure_all()
        self.theta = theta

        # Quanvolution filter (4‑wire)
        self.qfilter = QuanvolutionFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the quanvolution filter to the input tensor and return the
        flattened feature vector.  The output can be fed into a classical
        classifier downstream.
        """
        return self.qfilter(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the fully connected quantum circuit for each parameter in
        `thetas` and return the expectation value as a 1‑D numpy array.
        """
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

__all__ = ["HybridQuantumLayer", "QuanvolutionFilter"]
