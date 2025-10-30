import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict


class QuantumKernel(tq.QuantumModule):
    """
    Implements a fixed quantum kernel based on a 4‑wire ansatz.
    The kernel value is the absolute overlap of the final state vector
    after encoding two data vectors with opposite sign parameters.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel value for a single pair (x, y).
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0]).unsqueeze(0)


class QLSTMQuantum(tq.QuantumModule):
    """
    LSTM cell where each gate is implemented via a small
    variational quantum circuit (four RX gates followed by a CNOT chain).
    """
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int):
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
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                tgt = 0 if wire == self.n_wires - 1 else wire + 1
                tqf.cnot(qdev, wires=[wire, tgt])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)

        self.lin_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.lin_forget(combined)))
            i = torch.sigmoid(self.input(self.lin_input(combined)))
            g = torch.tanh(self.update(self.lin_update(combined)))
            o = torch.sigmoid(self.output(self.lin_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class HybridQLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM that optionally augments each token with
    a quantum kernel feature computed against a learnable set of basis
    vectors.  When ``use_kernel=False`` it reduces to the classical
    version implemented in the ML module.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        num_basis: int = 8,
        n_qubits: int = 4,
        use_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_kernel = use_kernel
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if self.use_kernel:
            self.kernel_basis = nn.Parameter(
                torch.randn(num_basis, embedding_dim), requires_grad=True
            )
            self.kernel = QuantumKernel()

        self.lstm = QLSTMQuantum(embedding_dim, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: LongTensor (seq_len,) of token indices.
        Returns:
            Log‑softmax scores of shape (seq_len, tagset_size).
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, embedding_dim)

        if self.use_kernel:
            seq_len = embeds.size(0)
            kernel_feats = []
            for basis_vec in self.kernel_basis:
                k_vals = []
                for idx in range(seq_len):
                    k = self.kernel(embeds[idx].unsqueeze(0), basis_vec.unsqueeze(0))
                    k_vals.append(k)
                k_vals = torch.cat(k_vals, dim=0).unsqueeze(-1)  # (seq_len, 1)
                kernel_feats.append(k_vals)
            kernel_feats = torch.cat(kernel_feats, dim=-1)  # (seq_len, num_basis)
            inputs = torch.cat([embeds, kernel_feats], dim=-1)
        else:
            inputs = embeds

        inputs = inputs.unsqueeze(0)  # batch=1
        lstm_out, _ = self.lstm(inputs)  # (1, seq_len, hidden_dim)
        lstm_out = lstm_out.squeeze(0)  # (seq_len, hidden_dim)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM"]
