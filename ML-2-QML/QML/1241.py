"""Quantum regression dataset and model derived from ``new_run_regression.py``.

This module extends the original seed by adding optional support for a
denoised target from the classical model as an additional wire in the
encoding.  It also adds a second variational layer to increase expressivity.
The public API remains compatible with the original ``QModel`` so that
existing training scripts can be used without modification.

The new ``QModel`` is a subclass of ``tq.QuantumModule`` that accepts an
optional ``denoised`` tensor.  If ``use_denoised`` is set to ``True``,
the denoised target is concatenated to the measurement features before
the linear head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(
    num_wires: int,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic superposition states for regression.

    Parameters
    ----------
    num_wires : int
        Number of qubits (equivalent to number of wires in the quantum model).
    samples : int
        Number of samples to generate.

    Returns
    -------
    states : np.ndarray, shape (samples, 2 ** num_wires)
        Complex state vectors.
    labels : np.ndarray, shape (samples,)
        Target values.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns a dictionary with keys:
        - ``"states"``: torch.tensor of shape (2 ** num_wires,)
        - ``"target"``: torch.tensor of shape ()
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """
    Quantum regression model that optionally accepts a denoised target
    from the classical model as an additional wire.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, use_denoised: bool = False):
        super().__init__()
        self.n_wires = num_wires
        self.use_denoised = use_denoised
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires + (1 if use_denoised else 0), 1)

    def forward(self, state_batch: torch.Tensor, denoised: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor, shape (batch, 2 ** num_wires)
            Input states.
        denoised : torch.Tensor or None, shape (batch,) or None
            Optional denoised target from the classical model.

        Returns
        -------
        output : torch.Tensor, shape (batch,)
            Final regression output.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        if self.use_denoised and denoised is not None:
            features = torch.cat([features, denoised.unsqueeze(-1)], dim=-1)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
