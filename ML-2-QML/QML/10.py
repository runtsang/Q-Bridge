import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMGen(nn.Module):
    """
    Quantum‑enhanced LSTM cell implemented with Pennylane.  Each gate
    is a variational circuit on ``n_qubits`` qubits.  The class
    mirrors the API of the classical counterpart, enabling
    direct comparison of performance on a quantum device or a
    simulator.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        device_name: str = "default.qubit",
        shots: int = 1024,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input embeddings.
        hidden_dim : int
            Dimensionality of the hidden state.
        n_qubits : int
            Number of qubits per gate.
        device_name : str, optional
            Pennylane device name.  Default is a simulator.
        shots : int, optional
            Number of shots for probability estimation.  Default 1024.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.device = qml.device(device_name, wires=n_qubits, shots=shots)

        # Linear projections for each gate
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Variational circuits for gates
        self.forget_circuit = self._make_circuit()
        self.input_circuit = self._make_circuit()
        self.update_circuit = self._make_circuit()
        self.output_circuit = self._make_circuit()

    def _make_circuit(self):
        @qml.qnode(self.device, interface="torch")
        def circuit(params: torch.Tensor):
            """
            Simple parameter‑shuffled rotation circuit.
            """
            for i, w in enumerate(range(self.n_qubits)):
                qml.RX(params[w], wires=w)
                qml.RZ(params[(w + 1) % self.n_qubits], wires=w)
                if w < self.n_qubits - 1:
                    qml.CNOT(wires=[w, w + 1])
                else:
                    qml.CNOT(wires=[w, 0])
            return qml.probs(wires=range(self.n_qubits))
        return circuit

    def _quantum_gate(self, linear: nn.Linear, circuit: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear projection followed by a variational circuit.
        """
        params = linear(x)  # (batch, n_qubits)
        probs = circuit(params)  # (batch, 2**n_qubits)
        # Reduce to n_qubits probabilities by taking first n_qubits entries
        # and normalising.
        probs = probs[:, :self.n_qubits]
        probs = probs / probs.sum(dim=1, keepdim=True)
        # Map to [0,1] using sigmoid for gates that require it
        return probs

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the quantum LSTM.

        Parameters
        ----------
        inputs : torch.Tensor
            Input sequence of shape (seq_len, batch, input_dim).
        states : tuple of torch.Tensor, optional
            Tuple (h_0, c_0) of initial hidden and cell states.
            Each of shape (batch, hidden_dim).  If None, zeros are used.

        Returns
        -------
        tuple
            - outputs : torch.Tensor of shape (seq_len, batch, hidden_dim)
            - (h_n, c_n) : final hidden and cell states.
        """
        h, c = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, h], dim=1)  # (batch, input+hidden)
            f = self._quantum_gate(self.forget_lin, self.forget_circuit, combined)
            i = self._quantum_gate(self.input_lin, self.input_circuit, combined)
            g = torch.tanh(self._quantum_gate(self.update_lin, self.update_circuit, combined))
            o = self._quantum_gate(self.output_lin, self.output_circuit, combined)

            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (h, c)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

__all__ = ["QLSTMGen"]
