"""Quantum hybrid kernel‑LSTM architecture for end‑to‑end sequence modeling."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["HybridKernelLSTM"]


class _RBFKernel(nn.Module):
    """Fast analytic RBF kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return exp(-gamma * ||x - y||^2)."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class _QuantumKernel(nn.Module):
    """Variational quantum kernel based on RX gates and a CNOT ladder."""

    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
                {"input_idx": [2], "func": "rx", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        self.param_layers = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
        )

    def _encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode a batch of data vectors and return the full state vector."""
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=data.shape[0], device=data.device)
        self.encoder(qdev, data)
        for i, gate in enumerate(self.param_layers):
            gate(qdev, wires=i)
        for i in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[i, i + 1])
        return qdev.states  # shape [batch, 2**n_wires]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the overlap |<ψ_x|ψ_y>| for each pair of vectors."""
        states_x = self._encode(x)
        states_y = self._encode(y)
        return torch.abs(torch.matmul(states_x, states_y.t()))


class QLSTM(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for i, gate in enumerate(self.params):
                gate(qdev, wires=i)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class HybridKernelLSTM(nn.Module):
    """
    Hybrid kernel‑LSTM that combines a classical RBF kernel,
    a variational quantum kernel, and a shared LSTM encoder
    that can use quantum gates.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vectors.
    hidden_dim : int
        Hidden size of the LSTM encoder.
    rbf_gamma : float, default 1.0
        Width parameter for the RBF kernel.
    n_qubits : int, default 4
        Number of qubits for the quantum kernel and quantum LSTM gates.
    support_vectors : torch.Tensor, optional
        Tensor of shape [num_support, input_dim] used to compute the kernel
        Gram matrix.  If ``None`` no kernel computation is performed.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rbf_gamma: float = 1.0,
        n_qubits: int = 4,
        support_vectors: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.rbf_kernel = _RBFKernel(gamma=rbf_gamma)
        self.q_kernel = _QuantumKernel(n_wires=n_qubits)
        self.lstm = QLSTM(input_dim, hidden_dim, n_qubits=n_qubits)
        self.support_vectors = support_vectors

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape [seq_len, batch, input_dim].

        Returns
        -------
        lstm_out : torch.Tensor
            LSTM hidden states of shape [seq_len, batch, hidden_dim].
        kernel_dict : dict
            Dictionary containing the classical RBF kernel matrix under
            key ``'rbf'`` and the quantum kernel matrix under
            key ``'quantum'``.  If no support vectors were supplied,
            an empty dictionary is returned.
        """
        seq_len, batch, _ = x.shape
        x_flat = x.reshape(seq_len * batch, -1)

        kernel_dict = {}
        if self.support_vectors is not None:
            sv = self.support_vectors.to(x.device)
            # Classical RBF kernel
            diff = x_flat.unsqueeze(1) - sv.unsqueeze(0)  # [N, M, D]
            kernel_dict["rbf"] = torch.exp(
                -self.rbf_kernel.gamma * torch.sum(diff * diff, dim=-1)
            )
            # Quantum kernel
            kernel_dict["quantum"] = self.q_kernel(x_flat, sv)

        lstm_out, _ = self.lstm(x)
        return lstm_out, kernel_dict
