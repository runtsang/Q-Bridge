import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumCircuitBuilder:
    """
    Builds a depth‑controlled variational ansatz that matches a classical
    feed‑forward network in terms of parameter count.
    """
    def __init__(self, num_qubits: int, depth: int = 3):
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights = self._build_circuit()

    def _build_circuit(self):
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        for val, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(val, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        return qc, encoding, weights

    def get_circuit(self) -> QuantumCircuit:
        return self.circuit

    def get_parameters(self):
        return self.encoding, self.weights

class QuantumHead(nn.Module):
    """
    Applies the variational circuit to classical feature embeddings
    and returns expectation values of local Z observables.
    """
    def __init__(self, num_qubits: int, depth: int = 3):
        super().__init__()
        self.num_qubits = num_qubits
        self.builder = QuantumCircuitBuilder(num_qubits, depth)
        self.observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                            for i in range(num_qubits)]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: tensor of shape (batch, num_qubits)
        Returns expectation values of shape (batch, num_qubits)
        """
        qc = self.builder.get_circuit()
        # Bind parameters
        bound_qc = qc.bind_parameters({f"x{i}": val.item() for i, val in enumerate(features.t())})
        state = Statevector.from_instruction(bound_qc)
        exp_vals = torch.tensor([state.expectation_value(obs).real for obs in self.observables])
        return exp_vals

class QuantumLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell using torchquantum modules.
    Mirrors the structure from the reference pair but is encapsulated
    for easy integration.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
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

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

class HybridModel(nn.Module):
    """
    Combines the classical backbone with the quantum head.
    """
    def __init__(self,
                 num_features: int,
                 depth: int = 3,
                 num_qubits: int = 4,
                 num_classes: int = 2):
        super().__init__()
        # Classical backbone
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        self.backbone = nn.Sequential(*layers)
        self.classical_head = nn.Linear(num_features, num_classes)
        self.quantum_head = QuantumHead(num_qubits, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if x.shape[1]!= self.quantum_head.num_qubits:
            raise ValueError("Feature dimension must match number of qubits for quantum head.")
        q_out = self.quantum_head(features)
        logits = self.classical_head(q_out)
        return logits

__all__ = ["QuantumCircuitBuilder", "QuantumHead", "QuantumLSTM", "HybridModel"]
