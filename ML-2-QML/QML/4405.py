"""QuantumHybridUnifiedModel – quantum‑centric implementation using TorchQuantum.

This module implements the same API as the classical version but all
sub‑modules are quantum.  The quantum version is useful for research
experiments where the forward pass is executed on a quantum simulator
or a real device via the TorchQuantum backend.
"""

import math
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Quantum convolution / patch extraction
# --------------------------------------------------------------------------- #
class QuantumConvPatchExtractor(tq.QuantumModule):
    """Quantum patch extractor that applies a random quantum kernel to every
    2×2 patch of the input image.
    """
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        bsz, _, h, w = x.shape
        patches = []
        for r in range(0, h, 2):
            for c in range(0, w, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)  # shape: (batch, 4 * num_patches)


# --------------------------------------------------------------------------- #
# 2. Quantum LSTM implementation
# --------------------------------------------------------------------------- #
class QLSTM(tq.QuantumModule):
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
            qdev = tq.QuantumDevice(self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
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

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# --------------------------------------------------------------------------- #
# 3. Quantum Transformer implementation
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(tq.QuantumModule):
    """Multi‑head attention that maps projections through quantum modules."""
    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                    {"input_idx": [4], "func": "rx", "wires": [4]},
                    {"input_idx": [5], "func": "rx", "wires": [5]},
                    {"input_idx": [6], "func": "rx", "wires": [6]},
                    {"input_idx": [7], "func": "rx", "wires": [7]},
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.q_layer = self.QLayer()
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        batch_size, seq_len, _ = x.size()
        # Split into heads
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_len, d_k)
        proj = []
        for head in range(self.num_heads):
            head_x = x[:, head, :, :].contiguous()  # (batch, seq_len, d_k)
            head_out = []
            for token in head_x.unbind(dim=1):
                qdev = tq.QuantumDevice(self.q_layer.n_wires, bsz=token.size(0), device=token.device)
                head_out.append(self.q_layer(token, qdev))
            head_out = torch.stack(head_out, dim=1)  # (batch, seq_len, d_k)
            proj.append(head_out)
        proj = torch.stack(proj, dim=1)  # (batch, heads, seq_len, d_k)
        proj = proj.transpose(1, 2).contiguous()  # (batch, seq_len, heads, d_k)
        proj = proj.view(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(proj)


class FeedForwardQuantum(tq.QuantumModule):
    """Feed‑forward network realised by a quantum module."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [idx], "func": "rx", "wires": [idx]}
                    for idx in range(n_qubits)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.linear1(out)
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(tq.QuantumModule):
    """Quantum‑enhanced transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# 4. Hybrid classifier head
# --------------------------------------------------------------------------- #
class HybridClassifier(tq.QuantumModule):
    """Linear classification head."""
    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# --------------------------------------------------------------------------- #
# 5. Unified quantum model
# --------------------------------------------------------------------------- #
class QuantumHybridUnifiedModel(tq.QuantumModule):
    """Unified model that stitches together the encoder, head and optional
    quantum sub‑modules.  The public API mirrors the original `QuantumNAT`
    and `Quanvolution` classes so that training scripts can call
    `model = QuantumHybridUnifiedModel(..., use_quantum_kernel=True)` etc.
    """
    def __init__(
        self,
        *,
        in_channels: int = 1,
        embed_dim: int = 64,
        num_classes: int = 10,
        encoder_mode: str = "lstm",
        use_quantum_kernel: bool = True,
        n_qubits: int = 4,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = QuantumConvPatchExtractor(in_channels)
        self.h_encoder = self._build_encoder(embed_dim, encoder_mode, n_qubits, n_layers)
        self.classifier = HybridClassifier(embed_dim, num_classes)

    def _build_encoder(self, embed_dim: int, mode: str, n_qubits: int, n_layers: int):
        mode = mode.lower()
        if mode == "lstm":
            return QLSTM(embed_dim, embed_dim, n_qubits)
        elif mode == "transformer":
            return nn.Sequential(
                *[TransformerBlockQuantum(
                    embed_dim,
                    num_heads=8,
                    ffn_dim=4 * embed_dim,
                    n_qubits=n_qubits
                ) for _ in range(n_layers)]
            )
        else:
            raise ValueError(f"Unsupported encoder mode {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quantum patch extraction
        patch_emb = self.encoder(x)  # shape: (batch, 4 * H/2 * W/2)
        # 2. Reshape to sequence for the encoder
        seq_len = patch_emb.shape[1] // self.embed_dim
        if seq_len * self.embed_dim!= patch_emb.shape[1]:
            raise ValueError("Patch embedding size not divisible by embed_dim")
        patch_seq = patch_emb.view(x.size(0), seq_len, self.embed_dim)
        # 3. Encode with hybrid LSTM/Transformer
        if isinstance(self.h_encoder, QLSTM):
            encoded, _ = self.h_encoder(patch_seq.permute(1, 0, 2))
        else:
            encoded = self.h_encoder(patch_seq)
        # 4. Classifier head
        return self.classifier(encoded)
