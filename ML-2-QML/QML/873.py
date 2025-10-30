import torch
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn as nn

class QuantumHiddenState(tq.QuantumModule):
    """
    Variational quantum circuit that maps the concatenated hidden and cell
    states (h, c) to a new hidden state h'.  The circuit encodes the
    classical vectors into qubits, applies trainable rotations and
    entangling gates, then measures all qubits to obtain a classical
    representation.  The output is sliced to match the hidden dimension.
    """
    def __init__(self, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        assert 2 * hidden_dim <= n_qubits, \
            "Number of qubits must be at least twice the hidden dimension"

        # Encoder: map (h, c) to the first 2*hidden_dim qubits via RX rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(2 * hidden_dim)
            ]
        )

        # Parameterised rotation layer (one RX,RY,RZ per qubit)
        self.param_rot = nn.Parameter(torch.randn(n_qubits, 3))

        # Measurement of all qubits in the Pauli‑Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, hidden_dim)
            c: (batch, hidden_dim)
        Returns:
            new_h: (batch, hidden_dim)
        """
        batch_size = h.size(0)
        device = h.device
        qdev = tq.QuantumDevice(n_wires=self.n_qubits,
                                bsz=batch_size,
                                device=device)

        # Encode (h, c) into the first 2*hidden_dim qubits
        enc_input = torch.cat([h, c], dim=1)
        self.encoder(qdev, enc_input)

        # Apply trainable rotations
        for i in range(self.n_qubits):
            tqf.rx(qdev, wires=i, params=self.param_rot[i, 0])
            tqf.ry(qdev, wires=i, params=self.param_rot[i, 1])
            tqf.rz(qdev, wires=i, params=self.param_rot[i, 2])

        # Entangling layer (CNOT chain)
        for i in range(self.n_qubits - 1):
            tqf.cnot(qdev, wires=[i, i + 1])

        # Measure all qubits
        out = self.measure(qdev)
        # Return only the first hidden_dim qubits as the new hidden state
        return out[:, :self.hidden_dim]

__all__ = ["QuantumHiddenState"]
