"""Quantum‑enhanced LSTM with kernelised gates.

The quantum variant mirrors the classical version but replaces
each linear gate with a small parameterised quantum circuit.
The state of the circuit is prepared through a quantum kernel
ansatz that embeds both the current input/hidden vector and a
learned kernel centre.  The overlap amplitude of the prepared
state is used as a feature for the gate, subsequently mapped to
the gate dimension via a tiny linear layer.

Author: gpt-oss-20b
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel that evaluates similarity between an input vector
    and a set of learnable kernel centres using a fixed circuit ansatz.
    """

    def __init__(self, n_wires: int, dim: int, num_centres: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.dim = dim
        # Initialise centres randomly
        init_centres = torch.randn(num_centres, dim) * 0.1
        self.centres = nn.Parameter(init_centres)
        # Define a simple circuit that can encode a vector of length dim
        # using RX rotations followed by a chain of CNOTs.
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(min(dim, n_wires))
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, q_dev: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode ``x`` forward and ``y`` backwards."""
        q_dev.reset_states(x.shape[0])
        # Forward encoding of x
        for info in self.encoder.op_list:
            func_name_dict[info["func"]](q_dev, wires=info["wires"], params=x[:, info["input_idx"]])
        # Backward encoding of y (inverse rotation)
        for info in reversed(self.encoder.op_list):
            func_name_dict[info["func"]](q_dev, wires=info["wires"], params=-y[:, info["input_idx"]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute kernel feature vector for each sample in ``x``."""
        batch_size = x.shape[0]
        device = x.device
        results = []
        for centre in self.centres:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch_size, device=device)
            self.forward(qdev, x, centre)
            # Overlap amplitude as kernel value
            results.append(torch.abs(qdev.states.view(-1)[0]))
        return torch.stack(results, dim=1)  # (B, K)

class HybridQLSTM(nn.Module):
    """Quantum LSTM cell where each gate is realised by a
    kernelised quantum circuit.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        num_kernels: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.num_kernels = num_kernels

        # Quantum kernel modules per gate
        self.kernel_f = QuantumKernel(n_qubits, input_dim + hidden_dim, num_kernels)
        self.kernel_i = QuantumKernel(n_qubits, input_dim + hidden_dim, num_kernels)
        self.kernel_g = QuantumKernel(n_qubits, input_dim + hidden_dim, num_kernels)
        self.kernel_o = QuantumKernel(n_qubits, input_dim + hidden_dim, num_kernels)

        # Linear heads to map kernel features to gate space
        self.linear_f = nn.Linear(num_kernels, hidden_dim)
        self.linear_i = nn.Linear(num_kernels, hidden_dim)
        self.linear_g = nn.Linear(num_kernels, hidden_dim)
        self.linear_o = nn.Linear(num_kernels, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, c = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, h], dim=1)  # (B, I+H)
            # Kernel features via quantum circuits
            k_f = self.kernel_f(combined)
            k_i = self.kernel_i(combined)
            k_g = self.kernel_g(combined)
            k_o = self.kernel_o(combined)

            # Gate activations
            f = torch.sigmoid(self.linear_f(k_f))
            i = torch.sigmoid(self.linear_i(k_i))
            g = torch.tanh(self.linear_g(k_g))
            o = torch.sigmoid(self.linear_o(k_o))

            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # (T, B, H)
        return outputs, (h, c)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between the quantum
    HybridQLSTM and a classical ``nn.LSTM``.

    Parameters
    ----------
    embedding_dim : int
    hidden_dim : int
    vocab_size : int
    tagset_size : int
    use_quantum : bool, default=False
        If ``True`` use the quantum HybridQLSTM, otherwise use the
        standard ``nn.LSTM``.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        use_quantum: bool = False,
        n_qubits: int = 4,
        num_kernels: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if use_quantum:
            self.lstm = HybridQLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                num_kernels=num_kernels,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)  # (T, B, E)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)  # (T, B, tagset)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["HybridQLSTM", "LSTMTagger"]
