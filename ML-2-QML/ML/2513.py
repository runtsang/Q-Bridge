import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class UnifiedSelfAttentionQLSTM(nn.Module):
    """
    Hybrid module that combines a multi‑head self‑attention block with a
    quantum‑enhanced LSTM.  The attention block is fully differentiable
    and can be swapped for a Qiskit circuit when the ``use_qiskit_attention``
    flag is set.  The LSTM part uses TorchQuantum gates for the gate
    computations, mirroring the QLSTM implementation from the reference
    pair.  The module is designed to be drop‑in compatible with the
    original SelfAttention and QLSTM classes.
    """
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 num_heads: int = 4,
                 use_qiskit_attention: bool = False,
                 n_qubits: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.num_heads = num_heads
        self.use_qiskit_attention = use_qiskit_attention

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Attention block
        if use_qiskit_attention:
            from.qml_code import QSelfAttention  # lazy import
            self.attention = QSelfAttention(embed_dim)
        else:
            self.attention = self._MultiHeadAttention(embed_dim, num_heads)

        # LSTM
        if n_qubits > 0:
            self.lstm = self._QLSTM(embed_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Output projection
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    # ------------------------------------------------------------------
    #  Classical multi‑head attention
    # ------------------------------------------------------------------
    class _MultiHeadAttention(nn.Module):
        def __init__(self, embed_dim: int, num_heads: int):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, D = x.shape
            q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
            return self.out_proj(out)

    # ------------------------------------------------------------------
    #  Quantum‑enhanced LSTM
    # ------------------------------------------------------------------
    class _QLSTM(nn.Module):
        class _QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
                self.encoder(qdev, x)
                for wire, gate in enumerate(self.params):
                    gate(qdev, wires=wire)
                for wire in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[wire, wire + 1])
                tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
                return self.measure(qdev)

        def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.n_qubits = n_qubits

            self.forget = self._QLayer(n_qubits)
            self.input_gate = self._QLayer(n_qubits)
            self.update = self._QLayer(n_qubits)
            self.output_gate = self._QLayer(n_qubits)

            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        def forward(self, inputs: torch.Tensor, states: tuple = None):
            hx, cx = self._init_states(inputs, states)
            outputs = []
            for x in inputs.unbind(dim=1):  # iterate over sequence
                combined = torch.cat([x, hx], dim=-1)
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outputs.append(hx.unsqueeze(1))
            out_seq = torch.cat(outputs, dim=1)
            return out_seq, (hx, cx)

        def _init_states(self, inputs: torch.Tensor, states: tuple = None):
            if states is not None:
                return states
            batch_size = inputs.size(0)
            device = inputs.device
            return (torch.zeros(batch_size, self.hidden_dim, device=device),
                    torch.zeros(batch_size, self.hidden_dim, device=device))

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        sentence: LongTensor of shape (B, T) containing token indices.
        Returns log‑softmax tag logits of shape (B, T, tagset_size).
        """
        # Embedding
        embeds = self.embedding(sentence)
        # Attention
        attn_out = self.attention(embeds)
        # LSTM
        lstm_out, _ = self.lstm(attn_out)
        # Tag projection
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["UnifiedSelfAttentionQLSTM"]
