import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit_machine_learning.neural_networks import NeuralNetworkEstimator
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

class QLayer(tq.QuantumModule):
    """Quantum gate layer used for LSTM gates."""
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
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires):
            if wire == self.n_wires - 1:
                tqf.cnot(qdev, wires=[wire, 0])
            else:
                tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)

class QuantumLSTM(nn.Module):
    """LSTM cell where gates are realized by small quantum circuits."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in torch.unbind(inputs, dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

def EstimatorQNN() -> nn.Module:
    """Quantum EstimatorQNN mirroring the classical version."""
    return QEstimatorQNN(
        circuit=QuantumCircuit(1),
        input_params=[ParameterVector("input", 1)],
        weight_params=[ParameterVector("weight", 1)],
        estimator=StatevectorEstimator(),
        observables=[SparsePauliOp.from_list([("Y", 1)])]
    )

def SamplerQNN() -> nn.Module:
    """Quantum SamplerQNN mirroring the classical version."""
    return QSamplerQNN(
        circuit=QuantumCircuit(2),
        input_params=ParameterVector("input", 2),
        weight_params=ParameterVector("weight", 4),
        sampler=StatevectorSampler()
    )

def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
    """Quantum circuit factory for the incremental data‑uploading classifier."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class QuantumClassifier(nn.Module):
    """Wrapper around a qiskit NeuralNetworkEstimator to produce logits."""
    def __init__(self, circuit: QuantumCircuit, encoding: list[ParameterVector], weights: list[ParameterVector], observables: list[SparsePauliOp]) -> None:
        super().__init__()
        self.estimator = NeuralNetworkEstimator(
            circuit=circuit,
            input_params=encoding,
            weight_params=weights,
            estimator=StatevectorEstimator()
        )
        self.output_layer = nn.Linear(len(observables), 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.estimator(x)
        return self.output_layer(out)

class HybridQLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM that combines quantum gates, a quantum estimator,
    a quantum sampler, and a quantum classifier. The interface matches the
    classical HybridQLSTM for side‑by‑side experiments.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_estimator: bool = True,
        use_sampler: bool = True,
        use_classifier: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = QuantumLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)

        self.use_estimator = use_estimator
        self.use_sampler = use_sampler
        self.use_classifier = use_classifier

        if use_estimator:
            self.estimator = EstimatorQNN()
        if use_sampler:
            self.sampler = SamplerQNN()
        if use_classifier:
            feature_dim = hidden_dim
            if use_estimator:
                feature_dim += 1
            if use_sampler:
                feature_dim += 2
            circuit, enc, wts, obs = build_classifier_circuit(feature_dim, depth=2)
            self.classifier = QuantumClassifier(circuit, enc, wts, obs)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of shape (seq_len, batch) containing word indices.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            log‑softmax logits from the quantum classifier and the tagger.
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed)
        lstm_out, _ = self.lstm(embeds)
        hidden_last = lstm_out[-1]  # (batch, hidden)
        features = hidden_last

        if self.use_estimator:
            est = self.estimator(features)  # (batch, 1)
            features = torch.cat([features, est], dim=1)
        if self.use_sampler:
            samp = self.sampler(features)  # (batch, 2)
            features = torch.cat([features, samp], dim=1)
        if self.use_classifier:
            logits = self.classifier(features)  # (batch, 2)
            tag_logits = self.hidden2tag(lstm_out[-1])  # (batch, tagset_size)
            return F.log_softmax(logits, dim=1), F.log_softmax(tag_logits, dim=1)
        else:
            return F.log_softmax(hidden_last, dim=1), None
