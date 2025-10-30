"""Hybrid quantum regression model with tunable depth and entanglement.

The quantum implementation mirrors the classical API but replaces the
feature extractor and head with a parameterised variational circuit.  The
circuit depth, entanglement strategy, and measurement basis are all
configurable, enabling systematic ablation studies.

Author: GPT-OSS-20B
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(
    num_wires: int,
    samples: int,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic quantum states of the form
    ``cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>`` and a smooth target.

    Parameters
    ----------
    num_wires : int
        Number of qubits.
    samples : int
        Number of samples.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    states : np.ndarray of shape (samples, 2**num_wires)
        Complex amplitude vectors.
    labels : np.ndarray of shape (samples,)
        Scalar target values.
    """
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
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset for quantum regression that returns complex state vectors
    and scalar targets.
    """

    def __init__(self, samples: int, num_wires: int, *, seed: int | None = None):
        self.states, self.labels = generate_superposition_data(num_wires, samples, seed=seed)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridRegression(tq.QuantumModule):
    """
    Quantum regression model with configurable feature map and variational depth.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the device.
    num_layers : int, optional
        Number of variational layers applied after the feature map.
        Defaults to 1.
    entanglement : str, optional
        Entanglement pattern for the variational layers.
        One of ``"cnot"`` (pairwise CNOT), ``"swap"`` (swap gates), or ``"full"``
        (all‑to‑all CNOT). Defaults to ``"cnot"``.
    """

    class _VariationalLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, entanglement: str):
            super().__init__()
            self.num_wires = num_wires
            self.entanglement = entanglement
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Random layer to break symmetry
            self.random = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))

        def forward(self, qdev: tq.QuantumDevice):
            self.random(qdev)
            # Apply single‑qubit rotations
            for w in range(self.num_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            # Entanglement
            if self.entanglement == "cnot":
                for w in range(self.num_wires - 1):
                    tq.CNOT()(qdev, wires=(w, w + 1))
            elif self.entanglement == "swap":
                for w in range(self.num_wires - 1):
                    tq.SWAP()(qdev, wires=(w, w + 1))
            elif self.entanglement == "full":
                for i in range(self.num_wires):
                    for j in range(i + 1, self.num_wires):
                        tq.CNOT()(qdev, wires=(i, j))

    def __init__(
        self,
        num_wires: int,
        num_layers: int = 1,
        entanglement: str = "cnot",
    ):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.entanglement = entanglement

        # Feature map: a simple angle‑encoded Ry circuit
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])

        # Variational layers
        self.v_layers = nn.ModuleList(
            [self._VariationalLayer(num_wires, entanglement) for _ in range(num_layers)]
        )

        # Measurement and classical head
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum feature map, variational layers,
        measurement, and classical head.

        Parameters
        ----------
        state_batch : torch.Tensor of shape (batch, 2**num_wires)
            Complex amplitude vectors representing input states.

        Returns
        -------
        torch.Tensor of shape (batch,)
            Predicted scalar values.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        for layer in self.v_layers:
            layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
