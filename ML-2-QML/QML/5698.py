import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding shared with the classical path."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class QuantumProjection(tq.QuantumModule):
    """
    Variational circuit that learns a linear projection of an input vector
    into a lower‑dimensional space.  The circuit is a simple chain of
    RX gates followed by a CNOT ladder.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(in_dim)]
        )
        self.gates = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(in_dim)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        qdev = tq.QuantumDevice(n_wires=self.in_dim, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        for i, gate in enumerate(self.gates):
            gate(qdev, wires=[i])
        # a simple CNOT ladder to entangle qubits
        for i in range(self.in_dim - 1):
            tqf.cnot(qdev, wires=[i, i + 1])
        return self.measure(qdev).view(bsz, self.out_dim)


class QuantumAttention(tq.QuantumModule):
    """
    Attention block where the linear projections of Q, K, V are replaced by
    variational circuits.  The soft‑max and subsequent weighting remain classical.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # classical linear layers for the final combination
        self.combine = nn.Linear(embed_dim, embed_dim)

        # quantum projection modules
        self.q_proj = QuantumProjection(embed_dim, self.d_k * num_heads)
        self.k_proj = QuantumProjection(embed_dim, self.d_k * num_heads)
        self.v_proj = QuantumProjection(embed_dim, self.d_k * num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        # quantum projections
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.combine(out)


class QuantumFeedForward(tq.QuantumModule):
    """
    Two‑layer feed‑forward block where the inner layer is a variational circuit.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.proj = QuantumProjection(embed_dim, n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(nn.Module):
    """Transformer block that uses quantum attention and feed‑forward."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim, num_heads, dropout)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_qubits, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class QLSTM(tq.QuantumModule):
    """LSTM cell where each gate is realised by a small variational circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # linear pre‑processing before the quantum gates
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # quantum circuits for the four gates
        self.forget_gate = QuantumProjection(n_qubits, n_qubits)
        self.input_gate = QuantumProjection(n_qubits, n_qubits)
        self.update_gate = QuantumProjection(n_qubits, n_qubits)
        self.output_gate = QuantumProjection(n_qubits, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq, _ = inputs.size()
        hx = torch.zeros(batch, self.hidden_dim, device=inputs.device)
        cx = torch.zeros(batch, self.hidden_dim, device=inputs.device)

        outputs = []
        for i in range(seq):
            x = inputs[:, i, :]
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, (hx, cx)


class HybridTextClassifier(nn.Module):
    """
    Quantum‑enhanced hybrid architecture that mirrors the classical API.
    Parameters that control the use of quantum sub‑modules are exposed.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_lstm: bool = False,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        n_qw: int = 0,  # number of qubits for quantum layers
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)

        self.use_lstm = use_lstm
        if use_lstm:
            if n_qw > 0:
                self.lstm_enc = QLSTM(embed_dim, lstm_hidden_dim, n_qw)
                lstm_out_dim = lstm_hidden_dim
            else:
                self.lstm_enc = nn.LSTM(
                    embed_dim, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True
                )
                lstm_out_dim = lstm_hidden_dim
        else:
            lstm_out_dim = embed_dim

        if n_qw > 0:
            # quantum transformer blocks
            self.transformer = nn.Sequential(
                *[
                    TransformerBlockQuantum(
                        embed_dim, num_heads, ffn_dim, n_qw, dropout
                    )
                    for _ in range(num_blocks)
                ]
            )
            transformer_out_dim = embed_dim
        else:
            # fall back to classical blocks
            self.transformer = nn.Sequential(
                *[
                    TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )
            transformer_out_dim = embed_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            transformer_out_dim if not use_lstm else lstm_out_dim,
            num_classes if num_classes > 2 else 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        if self.use_lstm:
            x, _ = self.lstm_enc(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = ["HybridTextClassifier"]
