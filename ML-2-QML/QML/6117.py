import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_quantum_classifier_circuit(num_qubits: int, depth: int):
    """
    Constructs a layered quantum ansatz for classification.
    Returns the circuit, the list of encoding parameters, the list of variational
    parameters, and the observable list used for measurement.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Data encoding layer
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational layers
    for layer in range(depth):
        for qubit, param in enumerate(weights[layer * num_qubits:(layer + 1) * num_qubits]):
            circuit.ry(param, qubit)
        # Entangling pattern
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
        circuit.cz(num_qubits - 1, 0)  # wrap‑around entanglement

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

def build_classical_classifier_circuit(num_features: int, depth: int):
    """
    Classical feed‑forward classifier mirroring the quantum implementation.
    Returns a nn.Sequential network, the list of input indices used for encoding,
    the list of weight‑size counters for each layer, and a placeholder list of
    observable indices.
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU(inplace=True)])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

class QuantumGateLayer(tq.QuantumModule):
    """
    Small quantum module used as a gate in the quantum LSTM.
    Encodes the classical input into qubit rotations, applies a trainable
    rotation on each wire, and produces a measurement that mimics a gate
    activation function.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
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
        # Entangle adjacent wires
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)

class QuantumQLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell where each gate is a small quantum module.
    The hidden state is still represented classically for compatibility with
    downstream PyTorch layers, but the gate activations are produced by quantum
    circuits that can be trained end‑to‑end.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_gate = QuantumGateLayer(n_qubits)
        self.input_gate = QuantumGateLayer(n_qubits)
        self.update_gate = QuantumGateLayer(n_qubits)
        self.output_gate = QuantumGateLayer(n_qubits)

        self.forget_fc = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_fc = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_fc = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_fc = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in torch.unbind(inputs, dim=1):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_fc(combined)))
            i = torch.sigmoid(self.input_gate(self.input_fc(combined)))
            g = torch.tanh(self.update_gate(self.update_fc(combined)))
            o = torch.sigmoid(self.output_gate(self.output_fc(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None):
        if states is not None:
            return states
        batch_size = inputs.size(0)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device)
        )

class HybridQLSTM(nn.Module):
    """
    Hybrid LSTM tagger that can operate with either a classical LSTM backbone
    or a quantum‑enhanced LSTM.  It also exposes a classifier head that can
    be either a classical feed‑forward network or a parameterised quantum
    ansatz, mirroring the structure in the classical branch.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 classifier_depth: int = 0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.n_qubits = n_qubits
        self.classifier_depth = classifier_depth

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        if n_qubits > 0:
            self.classifier_circuit, self.classifier_encoding, self.classifier_weights, self.classifier_observables = build_quantum_classifier_circuit(
                num_qubits=n_qubits, depth=classifier_depth
            )
        else:
            self.classifier, self.classifier_encoding, self.classifier_weight_sizes, self.classifier_observables = build_classical_classifier_circuit(
                num_features=hidden_dim, depth=classifier_depth
            )

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence tagging.  The LSTM may be quantum or classical,
        but the output always feeds a linear layer that produces tag logits.
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

    def get_classifier_metadata(self):
        """
        Return metadata that can be used by downstream quantum simulators
        or classical optimisers.
        """
        return self.classifier_encoding, self.classifier_weights, self.classifier_observables

__all__ = ["HybridQLSTM", "build_quantum_classifier_circuit", "build_classical_classifier_circuit"]
