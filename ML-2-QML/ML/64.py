import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ClassicalQLSTM(nn.Module):
    """Purely classical LSTM that mimics the original interface."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor,
                          Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outs = []
        for x in torch.unbind(inputs, dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outs.append(hx.unsqueeze(0))
        return torch.cat(outs, dim=0), (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        bs = inputs.size(1)
        return torch.zeros(bs, self.hidden_dim, device=inputs.device), \
               torch.zeros(bs, self.hidden_dim, device=inputs.device)

class HybridQLSTM(ClassicalQLSTM):
    """Hybrid LSTM where gate activations are produced by a shared
    variational quantum circuit.  The quantum circuit is executed once per
    timestep and its parameters are shared across the batch.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 quantum_backbone: callable):
        super().__init__(input_dim, hidden_dim)
        # `quantum_backbone` is a callable returning a torch Module
        # that maps a vector to a quantum‑derived gate activation.
        self.quantum_gate = quantum_backbone(input_dim + hidden_dim, n_qubits)
        self.use_quantum = True

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor,
                          Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outs = []
        for x in torch.unbind(inputs, dim=0):
            combined = torch.cat([x, hx], dim=1)
            # classical linear projections
            f_lin = torch.sigmoid(self.forget(combined))
            i_lin = torch.sigmoid(self.input_gate(combined))
            g_lin = torch.tanh(self.update(combined))
            o_lin = torch.sigmoid(self.output(combined))
            if self.use_quantum:
                # quantum‑derived activations
                q_act = self.quantum_gate(combined)
                f = f_lin * q_act
                i = i_lin * q_act
                g = g_lin * q_act
                o = o_lin * q_act
            else:
                f, i, g, o = f_lin, i_lin, g_lin, o_lin
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outs.append(hx.unsqueeze(0))
        return torch.cat(outs, dim=0), (hx, cx)

class QLSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical,
    hybrid, or fully quantum LSTM backbones."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 use_hybrid: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if use_hybrid and n_qubits > 0:
            self.lstm = HybridQLSTM(embedding_dim, hidden_dim,
                                    n_qubits,
                                    quantum_backbone=QuantumGateBlock)
        elif n_qubits > 0:
            self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

# Auxiliary quantum gate block (placeholder for a real variational circuit)
class QuantumGateBlock(nn.Module):
    """Simple variational circuit that maps a vector to a gate activation
    and can be replaced with a torch‑quantum implementation."""
    def __init__(self, input_dim: int, n_qubits: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, n_qubits),
            nn.ReLU(),
            nn.Linear(n_qubits, n_qubits),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
