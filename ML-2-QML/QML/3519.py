import torch
from torch import nn
from torch.nn import functional as F
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumRegressor(nn.Module):
    """
    Quantum-enhanced regression head using a single-qubit variational circuit.
    """
    def __init__(self):
        super().__init__()
        params = [Parameter(f"p{i}") for i in range(2)]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        self.estimator = QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[params[0]],
            weight_params=[params[1]],
            estimator=estimator
        )
        self.estimator.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x shape (batch, 1)
        np_x = x.detach().cpu().numpy().reshape(-1, 1)
        out = self.estimator.compute(np_x)
        return torch.tensor(out, dtype=torch.float32, device=x.device)

class QLayer(tq.QuantumModule):
    """
    Quantum layer used by the quantum LSTM gates.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ParameterList([nn.Parameter(torch.rand(1)) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for i, param in enumerate(self.params):
            tq.RX(param, wires=i)(qdev)
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)

class QLSTM(nn.Module):
    """
    Quantum LSTM cell where gates are realized by a shared quantum layer.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.qgate = QLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states=None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.qgate(self.linear_forget(combined)))
            i = torch.sigmoid(self.qgate(self.linear_input(combined)))
            g = torch.tanh(self.qgate(self.linear_update(combined)))
            o = torch.sigmoid(self.qgate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """
    Sequence tagger that can switch between classical and quantum LSTM.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

class HybridEstimator(nn.Module):
    """
    Hybrid model combining a quantum regressor and a quantum/classical tagger.
    """
    def __init__(self,
                 reg_input_dim: int = 2,
                 lstm_embed_dim: int = 50,
                 lstm_hidden_dim: int = 100,
                 vocab_size: int = 5000,
                 tagset_size: int = 10,
                 n_qubits: int = 0):
        super().__init__()
        self.regressor = QuantumRegressor()
        self.tagger = LSTMTagger(lstm_embed_dim,
                                 lstm_hidden_dim,
                                 vocab_size,
                                 tagset_size,
                                 n_qubits=n_qubits)

    def forward_regression(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)

    def forward_tagging(self, sentence: torch.Tensor) -> torch.Tensor:
        return self.tagger(sentence)

    def forward(self, x: torch.Tensor, mode: str = "regression") -> torch.Tensor:
        if mode == "regression":
            return self.forward_regression(x)
        elif mode == "tagging":
            return self.forward_tagging(x)
        else:
            raise ValueError(f"Unsupported mode {mode}")

__all__ = ["HybridEstimator", "LSTMTagger", "QuantumRegressor", "QLSTM", "QLayer"]
