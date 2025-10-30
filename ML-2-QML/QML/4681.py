"""Quantum implementation of the unified ConvGen127 model.

The QML version mirrors the classical architecture but replaces each component
with a parameterised quantum circuit.  It uses Qiskit for the filter and
classifier ansatzes and TorchQuantum for the regression head, enabling
gradient flow through quantum gates via the parameter‑shift rule.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
import torchquantum as tq

class ConvGen127(tq.QuantumModule):
    """
    Quantum counterpart of the classical ConvGen127.  The network consists of
    a parameterised filter circuit, an encoding circuit, and either a
    classification or regression head.

    Parameters
    ----------
    conv_kernel_size : int, default 2
        Number of qubits in the filter (kernel_size**2).
    shots : int, default 1024
        Number of shots per evaluation.
    depth : int, default 3
        Depth of the variational layers.
    task : str, {"classification", "regression"}, default "classification"
        Which downstream task to perform.
    """

    def __init__(
        self,
        conv_kernel_size: int = 2,
        shots: int = 1024,
        depth: int = 3,
        task: str = "classification",
    ) -> None:
        super().__init__()
        self.task = task
        self.shots = shots
        self.n_qubits = conv_kernel_size ** 2
        self.threshold = torch.tensor(127.0, dtype=torch.float32)  # emulation of classical threshold

        # ---------- Filter circuit ----------
        self.filter = self._build_filter_circuit()

        # ---------- Encoding circuit ----------
        self.encoding = ParameterVector("x", self.n_qubits)

        # ---------- Variational layers ----------
        self.weights = ParameterVector("theta", self.n_qubits * depth)
        self.circuit = self._build_classifier_circuit(depth)

        # ---------- Regression head ----------
        if task == "regression":
            self.head = nn.Linear(self.n_qubits, 1)

    def _build_filter_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Parameterised RX gates act as the data‑dependent encoding
        for i in range(self.n_qubits):
            qc.rx(self.encoding[i], i)
        # Randomised entangling layer
        qc.compose(random_circuit(self.n_qubits, depth=2), inplace=True)
        return qc

    def _build_classifier_circuit(self, depth: int) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Encode data
        for i, param in enumerate(self.encoding):
            qc.rx(param, i)
        # Variational layers
        idx = 0
        for _ in range(depth):
            for i in range(self.n_qubits):
                qc.ry(self.weights[idx], i)
                idx += 1
            # Entangling CZ chain
            for i in range(self.n_qubits - 1):
                qc.cz(i, i + 1)
        return qc

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Run the quantum circuit on a batch of classical states.

        Parameters
        ----------
        states : torch.Tensor
            Shape (B, N) where N = conv_kernel_size**2.

        Returns
        -------
        torch.Tensor
            Classification logits (shape (B, 2)) or regression output (shape (B,)).
        """
        bsz = states.shape[0]
        # Prepare a quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=states.device)

        # Encode data into the filter circuit
        self.filter(qdev, states)

        # Apply classifier ansatz
        self.circuit(qdev)

        # Measure all wires in Z basis
        meas = tq.MeasureAll(tq.PauliZ)
        meas(qdev)

        if self.task == "classification":
            # Two‑output logits: mean Z measurement and its complement
            logits = torch.stack([meas.results.mean(dim=1), 1 - meas.results.mean(dim=1)], dim=1)
            return logits
        else:
            # Regression head on the mean Z measurement
            features = meas.results.mean(dim=1).unsqueeze(-1)
            return self.head(features).squeeze(-1)

# --------------------------------------------------------------------------- #
# Dataset utilities (identical interface to the classical version)
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the superposition.
    samples : int
        Number of samples to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        State vectors (complex) and labels (float).
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.cos(thetas)[:, None] * np.eye(2 ** num_wires)[0] + \
        np.exp(1j * phis)[:, None] * np.sin(thetas)[:, None] * np.eye(2 ** num_wires)[-1]
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)

class SuperpositionDataset(torch.utils.data.Dataset):
    """
    Torch dataset that yields quantum state vectors and labels.
    """

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

__all__ = ["ConvGen127", "SuperpositionDataset", "generate_superposition_data"]
