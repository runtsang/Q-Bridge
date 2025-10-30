"""Quantum‑classical hybrid LSTM with quantum gates for each sub‑gate.
This version leverages Pennylane's device‑agnostic simulators and
tunes the autograd‑backpropagation via the parameter‑shifting rule.
"""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class HybridQLSTMCell(nn.Module):
    """
    A single LSTM cell that can replace each gate with either a quantum module
    or a classical linear module.  The quantum module is a small variational
    circuit implemented with Pennylane that outputs a single qubit per gate.
    The classical path uses a residual connection to the linear output so that
    the cell can be trained even if the quantum part is inactive.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        gate_name: str,
        *,
        use_quantum: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.gate_name = gate_name
        self.use_quantum = use_quantum

        # Linear projection for the gate
        self.linear = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum device
        self.device = qml.device("default.qubit", wires=n_qubits, shots=None)

        # Define a small variational circuit
        @qml.qnode(self.device, interface="torch", diff_method="parameter-shift")
        def circuit(params: torch.Tensor) -> torch.Tensor:
            for i in range(n_qubits):
                qml.RX(params[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.quantum_circuit = circuit

        # Residual weight
        self.residual_weight = nn.Parameter(torch.ones(n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear part
        lin_out = self.linear(x)
        # Quantum part
        if self.use_quantum:
            # The quantum circuit expects a vector of shape (n_qubits,)
            # For batch processing, we can loop over the batch dimension
            # but Pennylane supports batched inputs via the 'batch' argument.
            q_out = self.quantum_circuit(lin_out)
            # Combine with residual
            out = torch.sigmoid(lin_out + self.residual_weight * q_out)
        else:
            out = torch.sigmoid(lin_out)
        return out

class HybridQLSTM(nn.Module):
    """A full LSTM layer that uses HybridQLSTMCell for each gate."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        use_quantum_for: Optional[Tuple[str,...]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Define which gates have quantum support
        if use_quantum_for is None:
            use_quantum_for = ("forget", "input", "update", "output")
        self.gates = nn.ModuleDict({
            g: HybridQLSTMCell(
                input_dim,
                hidden_dim,
                n_qubits,
                gate_name=g,
                use_quantum=(g in use_quantum_for),
            )
            for g in ("forget", "input", "update", "output")
        })

        # Final linear for hidden state projection
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = self.gates["forget"](combined)
            i = self.gates["input"](combined)
            g = torch.tanh(self.gates["update"](combined))
            o = self.gates["output"](combined)
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

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

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = HybridQLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "LSTMTagger"]
