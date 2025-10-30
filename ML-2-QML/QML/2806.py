import torch
from torch import nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

class UnifiedQLayer(nn.Module):
    """
    Hybrid dense + quantum‑LSTM block.
    Parameters
    ----------
    input_dim : int
        Size of each time‑step input vector.
    hidden_dim : int
        Size of the hidden state that feeds the LSTM.
    n_qubits : int
        Number of qubits used in the variational circuit.
    use_quantum_gate : bool, default True
        Whether to use quantum gates (True) or fall back to classical
        linear gates (False). The classical fallback is provided for
        benchmarking.
    """
    class _QuantumGate(tq.QuantumModule):
        """Variational quantum circuit used as a gate activation."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Encode each input feature into a qubit using an RX rotation
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]}
                 for i in range(n_wires)]
            )
            # Trainable RX gates
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor):
            """
            Forward pass for the quantum gate.

            Parameters
            ----------
            x : torch.Tensor
                Shape (batch, n_wires).
            """
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires,
                bsz=x.shape[0],
                device=x.device,
            )
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # Entangling layer (CNOT chain)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int,
                 use_quantum_gate: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum_gate = use_quantum_gate

        # Dense mapping from input to hidden dimension
        self.dense = nn.Linear(input_dim, hidden_dim)

        # Linear layers to map combined input+hidden to n_qubits
        self.forget_lin = nn.Linear(hidden_dim * 2, n_qubits)
        self.input_lin = nn.Linear(hidden_dim * 2, n_qubits)
        self.update_lin = nn.Linear(hidden_dim * 2, n_qubits)
        self.output_lin = nn.Linear(hidden_dim * 2, n_qubits)

        if self.use_quantum_gate:
            self.forget_gate = self._QuantumGate(n_wires=n_qubits)
            self.input_gate = self._QuantumGate(n_wires=n_qubits)
            self.update_gate = self._QuantumGate(n_wires=n_qubits)
            self.output_gate = self._QuantumGate(n_wires=n_qubits)
        else:
            # Fallback to classical linear gates
            self.forget_gate = nn.Linear(n_qubits, n_qubits)
            self.input_gate = nn.Linear(n_qubits, n_qubits)
            self.update_gate = nn.Linear(n_qubits, n_qubits)
            self.output_gate = nn.Linear(n_qubits, n_qubits)

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        """
        Process a sequence of inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (seq_len, batch, input_dim).
        states : tuple[torch.Tensor, torch.Tensor] | None
            Optional initial hidden and cell states.
        """
        seq_len, batch, _ = inputs.shape
        hx, cx = self._init_states(batch, states)

        outputs = []
        for t in range(seq_len):
            x = inputs[t]
            # Dense mapping
            x_proj = self.dense(x)
            combined = torch.cat([x_proj, hx], dim=1)

            # Linear projection to n_qubits
            f_raw = self.forget_lin(combined)
            i_raw = self.input_lin(combined)
            g_raw = self.update_lin(combined)
            o_raw = self.output_lin(combined)

            # Quantum or classical gate activation
            f = torch.sigmoid(self.forget_gate(f_raw))
            i = torch.sigmoid(self.input_gate(i_raw))
            g = torch.tanh(self.update_gate(g_raw))
            o = torch.sigmoid(self.output_gate(o_raw))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, batch: int,
                     states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        device = next(self.parameters()).device
        hx = torch.zeros(batch, self.hidden_dim, device=device)
        cx = torch.zeros(batch, self.hidden_dim, device=device)
        return hx, cx

__all__ = ["UnifiedQLayer"]
