import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

class QuantumSelfAttention:
    """Quantum self‑attention block implemented with Qiskit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, rotation_params: np.ndarray,
            entangle_params: np.ndarray, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

class HybridQLSTMTagger(nn.Module):
    """Hybrid LSTMTagger that uses a quantum LSTM cell and a quantum
    self‑attention block.  The class mirrors the classical API so it can
    be swapped in seamlessly.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [0], "func": "rx", "wires": [0]},
                 {"input_idx": [1], "func": "rx", "wires": [1]},
                 {"input_idx": [2], "func": "rx", "wires": [2]},
                 {"input_idx": [3], "func": "rx", "wires": [3]}]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                    bsz=x.shape[0],
                                    device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # quantum LSTM gates
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)
        self.linear_forget = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # quantum attention
        self.attention = QuantumSelfAttention(n_qubits=n_qubits)
        self.backend = Aer.get_backend("qasm_simulator")

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def forward(self, sentence: torch.Tensor,
                rotation_params: np.ndarray | None = None,
                entangle_params: np.ndarray | None = None) -> torch.Tensor:
        embeds = self.word_embeddings(sentence).float()
        # default parameters if not supplied
        if rotation_params is None:
            rotation_params = np.zeros(self.attention.n_qubits * 3)
        if entangle_params is None:
            entangle_params = np.zeros(self.attention.n_qubits - 1)
        # quantum attention over embeddings
        counts = self.attention.run(self.backend,
                                    rotation_params,
                                    entangle_params,
                                    shots=1024)
        total_shots = sum(counts.values())
        probs = torch.zeros(self.attention.n_qubits,
                            device=embeds.device,
                            dtype=torch.float32)
        for bitstring, count in counts.items():
            bits = torch.tensor([int(b) for b in bitstring[::-1]],
                                dtype=torch.float32)
            probs += bits * count
        probs /= total_shots
        # broadcast attention vector to sequence length
        seq_len = embeds.size(0)
        attn_vec = probs.unsqueeze(0).expand(seq_len, -1)
        combined = torch.cat([embeds, attn_vec], dim=-1)
        hx, cx = self._init_states(combined, None)
        outputs = []
        for x in combined.unbind(dim=0):
            combined_x = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined_x)))
            i = torch.sigmoid(self.input(self.linear_input(combined_x)))
            g = torch.tanh(self.update(self.linear_update(combined_x)))
            o = torch.sigmoid(self.output(self.linear_output(combined_x)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        lstm_out = torch.cat(outputs, dim=0)
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTMTagger"]
