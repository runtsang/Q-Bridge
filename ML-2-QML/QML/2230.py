"""Hybrid quantum‑classical regressor.

This variant extends the classical MLP with a quantum post‑processing
module.  The quantum module can be configured as a single‑qubit
parameterised circuit or as a multi‑head quantum attention block
leveraging torchquantum.  The quantum part operates on the output of
the classical backbone and returns a scalar that is added to the
classical prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq

class HybridEstimatorQNN(nn.Module):
    """
    Quantum‑classical regression network.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input.
    hidden_dim : int, default 8
        Size of the hidden layer in the classical backbone.
    quantum_heads : int, default 1
        Number of quantum attention heads.  If 1, a single‑qubit
        parameterised circuit is used; if >1, a multi‑head quantum
        attention block is constructed.
    n_qubits : int, default 8
        Number of qubits per quantum head.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        quantum_heads: int = 1,
        n_qubits: int = 8,
    ) -> None:
        super().__init__()
        # Classical backbone identical to the pure ML version
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Quantum post‑processing
        self.quantum_heads = quantum_heads
        self.n_qubits = n_qubits
        if quantum_heads == 1:
            self.quantum_module = self._SingleQubitModule()
        else:
            self.quantum_module = self._MultiHeadAttentionModule(
                heads=quantum_heads, n_qubits=n_qubits
            )

    # ------------------------------------------------------------------ #
    #  Classical backbone
    # ------------------------------------------------------------------ #
    def _classical_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ------------------------------------------------------------------ #
    #  Quantum modules
    # ------------------------------------------------------------------ #
    class _SingleQubitModule(nn.Module):
        """Parameterised single‑qubit circuit that maps a scalar to a scalar."""

        def __init__(self) -> None:
            super().__init__()
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.rx = tq.RX(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            # x shape (batch, 1)
            q_device.reset()
            self.rz(q_device, wires=0, params=x.squeeze(-1))
            self.rx(q_device, wires=0)
            return self.measure(q_device).mean(dim=1, keepdim=True)

    class _MultiHeadAttentionModule(nn.Module):
        """Multi‑head quantum attention block that processes a batch of scalars."""

        def __init__(self, heads: int, n_qubits: int) -> None:
            super().__init__()
            self.heads = heads
            self.n_qubits = n_qubits
            self.q_layer = self._QLayer(n_qubits)
            self.combine = nn.Linear(heads, 1)

        class _QLayer(tq.QuantumModule):
            def __init__(self, n_qubits: int) -> None:
                super().__init__()
                self.n_wires = n_qubits
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [0], "func": "rx", "wires": [i]} for i in range(n_qubits)]
                )
                self.parameters = nn.ModuleList(
                    [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
                # Broadcast the scalar to all qubits
                batch = x.size(0)
                x_broadcast = x.repeat(1, self.n_wires)
                self.encoder(q_device, x_broadcast)
                for wire, gate in enumerate(self.parameters):
                    gate(q_device, wires=wire)
                return self.measure(q_device)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            # x shape (batch, 1)
            batch = x.size(0)
            outputs = []
            for i in range(self.heads):
                qdev = q_device.copy(bsz=batch, device=x.device)
                out = self.q_layer(x, qdev)  # shape (batch, n_qubits)
                out_mean = out.mean(dim=1, keepdim=True)  # (batch, 1)
                outputs.append(out_mean)
            out_stack = torch.stack(outputs, dim=1)  # (batch, heads, 1)
            out_stack = out_stack.squeeze(-1)  # (batch, heads)
            return self.combine(out_stack).unsqueeze(-1)

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted scalar of shape (batch, 1).
        """
        classical_out = self._classical_forward(x)
        # Prepare quantum device
        q_device = tq.QuantumDevice(n_wires=self.n_qubits)
        quantum_out = self.quantum_module(classical_out, q_device)
        return classical_out + quantum_out

__all__ = ["HybridEstimatorQNN"]
